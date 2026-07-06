extern crate alloc;

use core::marker::PhantomData;

use eerie::eerie_sys::runtime::{self as sys, iree_status_t};
use eerie::runtime::{
    self, api, base,
    base::StringView,
    hal::{self, BufferMapping, BufferView, EncodingType},
    vm::{DynamicList, List, Ref, ToRef},
};

include!(concat!(
    "../models/",
    env!("TASK"),
    "/",
    env!("MODEL"),
    "/size_",
    env!("MULT"),
    "/iree_build/iree_module_meta.rs"
));

// Init-array bounds the IREE runtime references but bare-metal startup lacks.
#[cfg(all(target_os = "none", not(feature = "std")))]
mod c_symbols {
    #[unsafe(no_mangle)]
    static __preinit_array_start: u8 = 0;
    #[unsafe(no_mangle)]
    static __preinit_array_end: u8 = 0;
    #[unsafe(no_mangle)]
    static __init_array_start: u8 = 0;
    #[unsafe(no_mangle)]
    static __init_array_end: u8 = 0;
}

unsafe extern "C" {
    fn module_create(
        instance: *const sys::iree_vm_instance_t,
        allocator: sys::iree_allocator_t,
        out_module: *mut *mut sys::iree_vm_module_t,
    ) -> iree_status_t;

    // Query fn of the model's statically-linked compute kernels (build.rs sets the name).
    #[link_name = env!("IREE_LIB_QUERY_FN_NAME")]
    fn iree_lib_query_fn(
        max_version: sys::iree_hal_executable_library_version_t,
        environment: *const sys::iree_hal_executable_environment_v0_t,
    ) -> *mut *const sys::iree_hal_executable_library_header_t;
}

// local-sync device whose executables come from the linked-in static library.
fn static_loader_device() -> hal::Device<'static> {
    let libraries: [sys::iree_hal_executable_library_query_fn_t; 1] = [Some(iree_lib_query_fn)];
    let host = base::Allocator::get_global().ctx;

    let mut params: sys::iree_hal_sync_device_params_t = Default::default();
    let ptr_params: *mut sys::iree_hal_sync_device_params_t = &mut params;
    unsafe { sys::iree_hal_sync_device_params_initialize(ptr_params) };
    params.arena_block_size = 4096;

    let import_provider: sys::iree_hal_executable_import_provider_t = Default::default();
    let mut loader = core::ptr::null_mut();
    base::Status::from_raw(unsafe {
        sys::iree_hal_static_library_loader_create(
            libraries.len(), libraries.as_ptr(), import_provider, host, &mut loader,
        )
    })
    .to_result()
    .expect("static_library_loader_create");

    let id = StringView::from("local-sync").ctx;
    let mut allocator = core::ptr::null_mut();
    base::Status::from_raw(unsafe {
        sys::iree_hal_allocator_create_heap(id, host, host, &mut allocator)
    })
    .to_result()
    .expect("allocator_create_heap");

    let mut device = core::ptr::null_mut();
    base::Status::from_raw(unsafe {
        sys::iree_hal_sync_device_create(id, ptr_params, 1, &mut loader, allocator, host, &mut device)
    })
    .to_result()
    .expect("sync_device_create");

    unsafe {
        sys::iree_hal_allocator_release(allocator);
        sys::iree_hal_executable_loader_release(loader);
    }
    hal::Device { ctx: device, marker: PhantomData }
}

pub fn predict_quantized<T, const C: usize>(input: &T) -> [f32; C] {
    let input = unsafe {
        core::slice::from_raw_parts(input as *const T as *const i8, core::mem::size_of_val(input))
    };
    let instance = api::Instance::new(
        &api::InstanceOptions::new(&mut runtime::hal::DriverRegistry::new())
            .use_all_available_drivers(),
    )
    .unwrap();
    let device = static_loader_device();
    let session = api::Session::create_with_device(
        &instance,
        &api::SessionOptions::default(),
        &device,
    )
    .unwrap();

    let module = unsafe {
        let mut out: *mut sys::iree_vm_module_t = core::ptr::null_mut();
        let status = module_create(
            sys::iree_runtime_instance_vm_instance(instance.ctx),
            base::Allocator::get_global().ctx,
            &mut out,
        );
        base::Status::from_raw(status)
            .to_result()
            .expect("module_create");
        out
    };
    unsafe { session.append_module(module.as_mut().unwrap()) }.unwrap();

    let function = session.lookup_function(ENTRY_POINT).unwrap();

    let input_list = DynamicList::<Ref<BufferView<i8>>>::new(1, &instance).unwrap();
    let in_buf = BufferView::<i8>::new(&session, INPUT_SHAPE, EncodingType::DenseRowMajor, input)
        .unwrap();
    let in_ref = in_buf.to_ref(&instance).unwrap();
    input_list.push_ref(&in_ref).unwrap();

    let output_list = DynamicList::<Ref<BufferView<i8>>>::new(1, &instance).unwrap();
    function.invoke(&input_list, &output_list).unwrap();

    let out_ref = output_list.get_ref(0).unwrap();
    let out_view: BufferView<i8> = out_ref.to_buffer_view(&session);
    let out_map = BufferMapping::new(out_view).unwrap();
    let out_data = out_map.data();

    let mut result = [0.0_f32; C];
    for c in 0..C {
        result[c] = (out_data[c] as f32 - OUTPUT_Q_ZP as f32) * OUTPUT_Q_SCALE;
    }
    result
}
