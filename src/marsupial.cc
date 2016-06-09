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

void GetRectangle(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = args.GetIsolate();

    dlib::rectangle a;
    a.set_top(1);
    a.set_left(1);
    a.set_right(11);
    a.set_bottom(11);

    Local<Object> obj = Object::New(isolate);
    translate_rectangle(a, obj, isolate);

    args.GetReturnValue().Set(obj);
}

//==============================================================
/**
 * Trainer
 */

// --- unpack records 
std::vector<TrainingRecord> unpack_traning_records(Isolate* isolate, Handle<Array>& tr_records) {
    std::vector<TrainingRecord> results;
    results.clear();

    for (int i = 0; i < tr_records->Length(); ++i) {
        const Handle<Object> js_record = Handle<Object>::Cast(tr_records->Get(i));
        TrainingRecord record;

        Handle<Value> imageFileName_value = js_record->Get(String::NewFromUtf8(isolate, "imageFileName"));
        String::Utf8Value imageFileName(imageFileName_value);
        record.imageFileName = std::string(*imageFileName);

        // Now for the matchAreas
        Handle<Array> matchAreas = Handle<Array>::Cast(js_record->Get(String::NewFromUtf8(isolate, "matchAreas")));
        record.matchAreas.clear();
        for (int j = 0; j < matchAreas->Length(); ++j) {
            Handle<Object> matchArea = Handle<Object>::Cast(matchAreas->Get(j));
            Handle<Value> left_val = matchArea->Get(String::NewFromUtf8(isolate, "left"));
            Handle<Value> top_val = matchArea->Get(String::NewFromUtf8(isolate, "top"));
            Handle<Value> width_val = matchArea->Get(String::NewFromUtf8(isolate, "width"));
            Handle<Value> height_val = matchArea->Get(String::NewFromUtf8(isolate, "height"));

            dlib::rectangle rect;            
            long width, height;
            rect.set_left(left_val->NumberValue());
            rect.set_top(top_val->NumberValue());
            width = width_val->NumberValue();
            height = height_val->NumberValue();
            rect.set_right(rect.left() + width);
            rect.set_bottom(rect.top() + height);

            record.matchAreas.push_back(rect);
        }

        results.push_back(record);
    }

    return results;
}

struct TrainWork {
    uv_work_t request;
    Persistent<Function> callback;

    std::vector<TrainingRecord> trainingRecords;
    std::string detectorOutputFileName;
    std::string error;
};

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

static void TrainComplete(uv_work_t* req, int status) {
    Isolate* isolate = Isolate::GetCurrent();

    v8::HandleScope handleScope(isolate);

    // Fire callback to signal the end
    TrainWork *work = static_cast<TrainWork *>(req->data);


    unsigned const argc = 1;
    Handle<Value> argv[argc] = { String::NewFromUtf8(isolate, work->error.c_str()) };
    Local<Function>::New(isolate, work->callback)->Call(isolate->GetCurrentContext()->Global(), argc, argv);

    work->callback.Reset();
    delete work;
}

static void TrainObjectDetector(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = args.GetIsolate();

    TrainWork* work = new TrainWork();
    work->request.data = work;

    if (args.Length() < 3) {
        isolate->ThrowException(Exception::TypeError(
            String::NewFromUtf8(isolate, "Wrong number of arguments")
        ));
        return;
    }
    
    Handle<Array> data = Handle<Array>::Cast(args[0]);
    String::Utf8Value detectorOutputFileName(args[1]->ToString());

    work->trainingRecords = unpack_traning_records(isolate, data);
    work->detectorOutputFileName = std::string(*detectorOutputFileName);
    work->error = "";

    // Store the callback
    Local<Function> callback = Local<Function>::Cast(args[2]);
    work->callback.Reset(isolate, callback);

    // Start the async process
    uv_queue_work(uv_default_loop(), &work->request, TrainAsync, TrainComplete);

    args.GetReturnValue().Set(Undefined(isolate));
}

// =======================================================================================
// Detector
//

struct DetectWork {
    uv_work_t request;
    Persistent<Function> callback;

    std::string imageFileName;
    std::string svmDetectorFileName;

    std::vector<rectangle> results;
    std::string error;
};

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

static void DetectObjects(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = args.GetIsolate();

    DetectWork* work = new DetectWork();
    work->request.data = work;

    if (args.Length() < 3) {
        isolate->ThrowException(Exception::TypeError(
            String::NewFromUtf8(isolate, "Wrong number of arguments")
        ));
        return;
    }

    String::Utf8Value imageFileName(args[0]->ToString());
    String::Utf8Value svmDetectorFileName(args[1]->ToString());

    work->imageFileName = std::string(*imageFileName);
    work->svmDetectorFileName = std::string(*svmDetectorFileName);

    // Store the callback
    Local<Function> callback = Local<Function>::Cast(args[2]);
    work->callback.Reset(isolate, callback);

    // Start the async process
    uv_queue_work(uv_default_loop(), &work->request, DetectAsync, DetectComplete);

    args.GetReturnValue().Set(Undefined(isolate));
}



void init(Local<Object> exports) {
    NODE_SET_METHOD(exports, "getRectangle", GetRectangle);
    NODE_SET_METHOD(exports, "trainObjectDetector", TrainObjectDetector);
    NODE_SET_METHOD(exports, "detectObjects", DetectObjects);
}

NODE_MODULE(recognition, init)


