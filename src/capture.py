import freenect

def get_depth():
    depth, _ = freenect.sync_get_depth(format=freenect.DEPTH_MM)
    return depth

def get_ir():
    ir, _ = freenect.sync_get_video(format=freenect.VIDEO_IR_8BIT)
    return ir

def depth_ir():
    return (get_depth(), get_ir())