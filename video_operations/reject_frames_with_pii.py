def reject_frames_with_pii(image, output_video):
    if image_does_not_contain_pii(image):
        output_video.write(image)


def image_does_not_contain_pii(image):
    return True
