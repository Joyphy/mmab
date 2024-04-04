backend_config = dict(
    common_config=dict(fp16_mode=False, max_workspace_size=1073741824),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    max_shape=[ 1, 3, 256, 256],
                    min_shape=[ 1, 3, 256, 256],
                    opt_shape=[ 1, 3, 256, 256]))),
    ],
    type='tensorrt')
codebase_config = dict(
    model_type='end2end',
    post_processing=dict(
        background_label_id=-1,
        confidence_threshold=0.005,
        iou_threshold=0.5,
        keep_top_k=100,
        max_output_boxes_per_class=200,
        pre_top_k=5000,
        score_threshold=0.05),
    task='ObjectDetection',
    type='mmdet')
onnx_config = dict(
    export_params=True,
    input_names=[
        'input',
    ],
    input_shape=(256, 256),
    keep_initializers_as_inputs=False,
    opset_version=11,
    optimize=True,
    output_names=[
        'dets',
        'labels',
        'masks'
    ],
    save_file='end2end.onnx',
    type='onnx')