// pmml_runtime.hpp — C++ RAII wrapper over pmml_runtime.h (like onnxruntime_cxx_api.h)
//
// Thin header-only wrapper. No extra .so. Throws pmml::Exception on PmmlStatus != OK.
// Usage:
//   pmml::Env env{PMML_LOG_WARNING, "app"};
//   pmml::SessionOptions so; so.SetGraphOptimizationLevel(PMML_GRAPH_ENABLE_BASIC);
//   pmml::Session sess{env, "model.pmml", so};
//   auto out = sess.Run({{"Petal.Length", 1.4}, {"Petal.Width", 0.2}});
//   std::cout << out.at("predictedValue").as_string(sess) << "\n";

#pragma once
#include "pmml_runtime.h"
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace pmml {

struct Exception : std::runtime_error {
    PmmlErrorCode code;
    Exception(PmmlErrorCode c, const std::string& msg) : std::runtime_error(msg), code(c) {}
};

inline void check(PmmlStatus* s, const PmmlApi* api) {
    if (!s) return;
    auto code = api->GetErrorCode(s);
    std::string msg = api->GetErrorMessage(s);
    api->ReleaseStatus(s);
    throw Exception(code, msg);
}

class Env {
public:
    explicit Env(PmmlLogLevel level = PMML_LOG_WARNING, const char* id = "pmml") {
        auto* api = PmmlGetApi(PMML_API_VERSION);
        check(api->CreateEnv(level, id, &env_), api);
        api_ = api;
    }
    ~Env() { if (env_) api_->ReleaseEnv(env_); }
    Env(const Env&) = delete;
    Env& operator=(const Env&) = delete;
    Env(Env&& o) noexcept : env_(o.env_), api_(o.api_) { o.env_ = nullptr; }
    PmmlEnv* get() const { return env_; }
    const PmmlApi* api() const { return api_; }
private:
    PmmlEnv* env_ = nullptr;
    const PmmlApi* api_ = nullptr;
};

class SessionOptions {
public:
    SessionOptions() {
        auto* api = PmmlGetApi(PMML_API_VERSION);
        check(api->CreateSessionOptions(&opts_), api);
        api_ = api;
    }
    ~SessionOptions() { if (opts_) api_->ReleaseSessionOptions(opts_); }
    void SetGraphOptimizationLevel(PmmlGraphOptimizationLevel lvl) { check(api_->SetGraphOptimizationLevel(opts_, lvl), api_); }
    void SetIntraOpNumThreads(int n) { check(api_->SetIntraOpNumThreads(opts_, n), api_); }
    void AppendExecutionProvider(const char* name) { check(api_->AppendExecutionProvider(opts_, name, nullptr, nullptr, 0), api_); }
    PmmlSessionOptions* get() const { return opts_; }
private:
    PmmlSessionOptions* opts_ = nullptr;
    const PmmlApi* api_ = nullptr;
};

class Session {
public:
    Session(const Env& env, const char* path, const SessionOptions& opts) : api_(env.api()) {
        check(api_->CreateSession(env.get(), path, opts.get(), &sess_), api_);
    }
    Session(const Env& env, const void* bytes, size_t len, const SessionOptions& opts) : api_(env.api()) {
        check(api_->CreateSessionFromArray(env.get(), bytes, len, opts.get(), &sess_), api_);
    }
    ~Session() { if (sess_) api_->ReleaseSession(sess_); }

    size_t GetInputCount() const { size_t n=0; check(api_->SessionGetInputCount(sess_, &n), api_); return n; }
    std::string GetInputName(size_t i) const { const char* s=nullptr; check(api_->SessionGetInputName(sess_, i, &s), api_); return s? s:""; }
    const char* GetVersionString() const { return api_->GetVersionString(); }

    // RowMajor single/batch via flat arrays — convenience wrapper
    std::unordered_map<std::string, PmmlValue> Run(const std::unordered_map<std::string, PmmlValue>& inputs) {
        // TODO: picks output_names from SessionGetOutputCount/Name; fills output flat and maps back
        // Stub for header illustration — real impl lives in c/src/
        (void)inputs;
        return {};
    }

private:
    PmmlSession* sess_ = nullptr;
    const PmmlApi* api_ = nullptr;
};

} // namespace pmml
