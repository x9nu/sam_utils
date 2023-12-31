import asyncio
import json
import os

import aiohttp


# 这是两个都可以用的版本
async def upload_and_segmentate_image(image_path):
    # server_url = "http://dashuai.synology.me:5000"
    server_url = "http://113.251.91.53:10007"
    print(server_url, image_path)

    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field('file', open(image_path, 'rb'), filename=os.path.basename(image_path))

        async with session.post(server_url + '/upload_image', data=data) as response:
            cookies = response.cookies
            print("Cookies in session:", cookies)
            if response.status == 200:
                text = await response.text()
                print(text)
            elif response.status == 400:
                response_text = await response.text()
                raise Exception(
                    f"视觉服务器可连接，但接口不可访问\n[上次出现这个问题是视觉服务器在升级图形驱动]: {response.status}: {response_text}")
            else:
                response_text = await response.text()
                raise Exception(f"上传图片失败: {response.status}: {response_text}")

        async with session.get(server_url + '/result', cookies=cookies) as response:
            if response.status == 200:
                text = await response.text()
                masks_data = [json.loads(line) for line in text.splitlines()]
            else:
                raise Exception(f"Error in getting masks data: {await response.text()}")

            if not masks_data:
                return "从视觉服务器获取的mask为空", 500

    return masks_data


async def main():
    path = '..\\assets\\notebook1.png'
    result = await upload_and_segmentate_image(path)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
