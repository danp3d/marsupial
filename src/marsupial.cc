#include <node.h>
#include <dlib/geometry.h>
#include "data_parser.h"
#include <v8.h>
#include <uv.h>
#include <string>
#include <iostream>
#include <vector>
#include <thread>
#include "trainer.h"
#include "detector.h"

using namespace v8;

//==============================================================
/**
 * Trainer
 */

// --- unpack records 
std::vector<TrainingRecord> unpack_traning_records(Isolate* isolate, Handle<Array>& tr_records) {
    std::vector<TrainingRecord> results;
    results.clear();

    // Go through all the records in the JS array
    for (int i = 0; i < tr_records->Length(); ++i) {
        // Get each item in the array as a JS object
        const Handle<Object> js_record = Handle<Object>::Cast(tr_records->Get(i));
        TrainingRecord record;

        // Get the 'imageFileName' property
        Handle<Value> imageFileName_value = js_record->Get(String::NewFromUtf8(isolate, "imageFileName"));
        String::Utf8Value imageFileName(imageFileName_value);
        record.imageFileName = std::string(*imageFileName);

        // Now for the matchAreas
        Handle<Array> matchAreas = Handle<Array>::Cast(js_record->Get(String::NewFromUtf8(isolate, "matchAreas")));
        record.matchAreas.clear();

        // For each item in the 'machAreas' array
        for (int j = 0; j < matchAreas->Length(); ++j) {
            // Get the object and all the values
            Handle<Object> matchArea = Handle<Object>::Cast(matchAreas->Get(j));
            Handle<Value> left_val = matchArea->Get(String::NewFromUtf8(isolate, "left"));
            Handle<Value> top_val = matchArea->Get(String::NewFromUtf8(isolate, "top"));
            Handle<Value> width_val = matchArea->Get(String::NewFromUtf8(isolate, "width"));
            Handle<Value> height_val = matchArea->Get(String::NewFromUtf8(isolate, "height"));

            // Store the values as a rectangle
            dlib::rectangle rect;            
            long width, height;
            rect.set_left(left_val->NumberValue());
            rect.set_top(top_val->NumberValue());
            width = width_val->NumberValue();
            height = height_val->NumberValue();
            rect.set_right(rect.left() + width - 1);
            rect.set_bottom(rect.top() + height - 1);

            record.matchAreas.push_back(rect);
        }

        results.push_back(record);
    }

    // Records unpacked - we've turned the JS objects into a vector<TrainingRecord>
    return results;
}

// Struct representing the async job of training an object detector
struct TrainWork {
    uv_work_t request;
    Persistent<Function> callback;

    std::vector<TrainingRecord> trainingRecords;
    std::string detectorOutputFileName;
    std::string error;
};

// The actual async job
static void TrainAsync(uv_work_t* req) {
    TrainWork* work = static_cast<TrainWork*>(req->data);

    try {
        train_object_detector(work->trainingRecords, work->detectorOutputFileName);
    }
    catch (std::exception& e) {
        work->error = e.what();
    }
    catch (dlib::error* e) {
        work->error = e->what();
    }
    catch (std::string& e) {
        work->error = e;
    }
    catch (...) {
        work->error = "Unknown exception happened";
    }
}

// Function to be called once the job is complete
static void TrainComplete(uv_work_t* req, int status) {
    Isolate* isolate = Isolate::GetCurrent();

    v8::HandleScope handleScope(isolate); // This is a weird one. This is required on node 4.*, but is a bit of a mistery.

    TrainWork *work = static_cast<TrainWork *>(req->data);

    // Fire callback to signal the end. Only one argument: the error parameter. If null or empty, it means it all worked fine.
    unsigned const argc = 1;
    Handle<Value> argv[argc] = { String::NewFromUtf8(isolate, work->error.c_str()) };
    Local<Function>::New(isolate, work->callback)->Call(isolate->GetCurrentContext()->Global(), argc, argv);

    // Free up the callback and work objects
    work->callback.Reset();
    delete work;
}

// Function called by the JS code
static void TrainObjectDetector(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = args.GetIsolate();

    TrainWork* work = new TrainWork();
    work->request.data = work;

    // Check arguments
    if (args.Length() < 3) {
        isolate->ThrowException(Exception::TypeError(
            String::NewFromUtf8(isolate, "Wrong number of arguments")
        ));
        return;
    }

    // Get argument values: argument 0 is an array of objects; argument 1 is a string with the detector name
    Handle<Array> data = Handle<Array>::Cast(args[0]);
    String::Utf8Value detectorOutputFileName(args[1]->ToString());

    // Unpack the values
    work->trainingRecords = unpack_traning_records(isolate, data);
    work->detectorOutputFileName = std::string(*detectorOutputFileName);
    work->error = "";

    // Store the callback
    Local<Function> callback = Local<Function>::Cast(args[2]);
    work->callback.Reset(isolate, callback);

    // Start the async process
    uv_queue_work(uv_default_loop(), &work->request, TrainAsync, TrainComplete);

    // Return undefined
    args.GetReturnValue().Set(Undefined(isolate));
}

// =======================================================================================
// Detector
//

// Work structure (needed by libuv)
struct DetectWork {
    uv_work_t request;
    Persistent<Function> callback;

    std::string imageFileName;
    std::string svmDetectorFileName;

    std::vector<rectangle> results;
    std::string error;
};

// Function that will be executed asynchronously (libuv takes charge of executing it)
static void DetectAsync(uv_work_t* req) {
    DetectWork* work = static_cast<DetectWork*>(req->data);

    try {
        work->results = detect_objects(work->imageFileName, work->svmDetectorFileName);
    }
    catch (std::exception& e) {
        work->error = e.what();
    }
    catch (dlib::error* e) {
        work->error = e->what();
    }
    catch (std::string& e) {
        work->error = e;
    }
    catch (...) {
        work->error = "Unknown exception happened";
    }
}

// Function that will be called once the asynchronous work has completed
static void DetectComplete(uv_work_t* req, int status) {
    Isolate* isolate = Isolate::GetCurrent();

    v8::HandleScope handleScope(isolate);
    DetectWork *work = static_cast<DetectWork*>(req->data);

    // Translate the vector<rectangle> into something v8 can understand
    Local<Array> result_list = Array::New(isolate);
    for (int i = 0; i < work->results.size(); i++) {
        Local<Object> result = Object::New(isolate);
        translate_rectangle(work->results[i], result, isolate);
        result_list->Set(i, result);
    }

    Local<String> error = String::NewFromUtf8(isolate, work->error.c_str());

    // Fire callback to signal the end
    unsigned const argc = 2;
    Handle<Value> argv[argc] = { error, result_list };
    Local<Function>::New(isolate, work->callback)->Call(isolate->GetCurrentContext()->Global(), argc, argv);

    work->callback.Reset();
    delete work;
}

// Function called by the JavaScript side. This will populate the Work struct and fire the async job
static void DetectObjects(const FunctionCallbackInfo<Value>& args) {
    // Node's heap implementation. We need this whenever we use variables from the JS side (or when we create variables that will be accessible to JS)
    Isolate* isolate = args.GetIsolate();

    DetectWork* work = new DetectWork();
    work->request.data = work;

    if (args.Length() < 3) {
        isolate->ThrowException(Exception::TypeError(
                    String::NewFromUtf8(isolate, "Wrong number of arguments")
                    ));
        return;
    }

    // Converting the arguments to String values
    String::Utf8Value imageFileName(args[0]->ToString());
    String::Utf8Value svmDetectorFileName(args[1]->ToString());

    work->imageFileName = std::string(*imageFileName);
    work->svmDetectorFileName = std::string(*svmDetectorFileName);

    // Convert the 3rd argument to a callback function and store it for later usage
    Local<Function> callback = Local<Function>::Cast(args[2]);
    work->callback.Reset(isolate, callback);

    // Start the async process
    uv_queue_work(uv_default_loop(), &work->request, DetectAsync, DetectComplete);

    // Return undefined
    args.GetReturnValue().Set(Undefined(isolate));
}

// =======================================================================================
// This section is the equivalent of module.exports in JS
//

void init(Local<Object> exports) {
    NODE_SET_METHOD(exports, "trainObjectDetector", TrainObjectDetector);
    NODE_SET_METHOD(exports, "detectObjects", DetectObjects);
}

NODE_MODULE(recognition, init)


