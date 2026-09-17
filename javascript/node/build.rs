fn main() {
    // napi-rs needs this to emit the export list a Node addon links against.
    // Without it, macOS fails to link `napi_*` symbols.
    napi_build::setup();
}
