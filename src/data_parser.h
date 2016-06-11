#include <node.h>
#include <dlib/geometry.h>
#include <vector>

using namespace v8;

// Translate a dlib::rectangle into a JS object
void translate_rectangle(dlib::rectangle& r, Local<Object> output, Isolate* isolate) {
    output->Set(String::NewFromUtf8(isolate, "top"), Number::New(isolate, r.top()));
    output->Set(String::NewFromUtf8(isolate, "left"), Number::New(isolate, r.left()));
    output->Set(String::NewFromUtf8(isolate, "width"), Number::New(isolate, r.width()));
    output->Set(String::NewFromUtf8(isolate, "height"), Number::New(isolate, r.height()));
}

