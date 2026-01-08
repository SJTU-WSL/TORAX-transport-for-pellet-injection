from PIL import Image
import os


def create_gif(output_name='animation.gif', duration=100):
    image_list = []

    # 按照 0 到 99 的顺序读取文件
    # 使用 range(100) 确保顺序是数值顺序 (0, 1, 2...) 而不是字符顺序 (0, 1, 10...)
    for i in range(100):
        filename = 's_face/' + f"{i}.png"

        # 检查文件是否存在，防止中间缺图导致报错
        if os.path.exists(filename):
            try:
                img = Image.open(filename)
                image_list.append(img)
            except IOError:
                print(f"无法读取文件: {filename}")
        else:
            print(f"文件跳过 (不存在): {filename}")

    if image_list:
        # 保存 GIF
        # duration: 每一帧的持续时间，单位是毫秒 (ms)。100ms = 10fps
        # loop: 0 表示无限循环，1 表示只播放一次
        image_list[0].save(
            output_name,
            save_all=True,
            append_images=image_list[1:],
            duration=duration,
            loop=0
        )
        print(f"成功生成: {output_name}，共包含 {len(image_list)} 帧。")
    else:
        print("未找到任何图片。")


if __name__ == "__main__":
    create_gif()