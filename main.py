import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
from collections import Counter
import heapq
import bitarray
import matplotlib.pyplot as plt
import os

### --- Цветовое преобразование --- ###

def rgb_to_ycbcr(image):
    matrix = np.array([[0.299, 0.587, 0.114],
                       [-0.169, -0.331, 0.5],
                       [0.5, -0.419, -0.081]])
    image = np.array(image)
    ycbcr = np.dot(image, matrix.T)
    ycbcr[..., 1:] += 128
    return ycbcr

def ycbcr_to_rgb(ycbcr):
    matrix = np.array([[1, 0, 1.402],
                       [1, -0.344136, -0.714136],
                       [1, 1.772, 0]])
    ycbcr = ycbcr.copy()
    ycbcr[..., 1:] -= 128
    rgb = np.dot(ycbcr, matrix.T)
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)

### --- Блоки и DCT --- ###
def dct_2d(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')
def idct_2d(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')
def quantize(dct_block, quant_matrix):
    return np.round(dct_block / quant_matrix)
def dequantize(quant_block, quant_matrix):
    return quant_block * quant_matrix
def split_to_blocks(image, block_size=8):
    height, width = image.shape
    blocks = []
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image[i:i + block_size, j:j + block_size]
            if block.shape == (block_size, block_size):  # Убедитесь, что блок имеет правильный размер
                blocks.append(block)
            else:
                print(f"Warning: Skipping block at ({i},{j}), invalid block size {block.shape}")
    return blocks

def merge_blocks(blocks, height, width, N):
    image = np.zeros((height, width))
    idx = 0
    for i in range(0, height, N):
        for j in range(0, width, N):
            image[i:i+N, j:j+N] = blocks[idx]
            idx += 1
    return image
### --- Зигзаг --- ###
def zigzag(block):
    index_order = sorted(((x, y) for x in range(8) for y in range(8)),
                         key=lambda s: (s[0]+s[1], -s[1] if (s[0]+s[1]) % 2 else s[1]))
    return np.array([block[x, y] for x, y in index_order])
def inverse_zigzag(arr):
    block = np.zeros((8, 8))
    index_order = sorted(((x, y) for x in range(8) for y in range(8)),
                         key=lambda s: (s[0]+s[1], -s[1] if (s[0]+s[1]) % 2 else s[1]))
    for idx, (x, y) in enumerate(index_order):
        block[x, y] = arr[idx]
    return block
### --- Хаффман --- ###
class Node:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None
    def __lt__(self, other): return self.freq < other.freq
def build_huffman_tree(data):
    counter = Counter(data)
    heap = [Node(sym, freq) for sym, freq in counter.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)
        parent = Node(None, n1.freq + n2.freq)
        parent.left, parent.right = n1, n2
        heapq.heappush(heap, parent)

    def build_table(node, prefix=""):
        if node.symbol is not None:
            return {node.symbol: prefix}
        left = build_table(node.left, prefix + "0")
        right = build_table(node.right, prefix + "1")
        return {**left, **right}

    return build_table(heap[0])
def huffman_encode(data, code_table):
    return ''.join(code_table[s] for s in data)
def huffman_decode(bitstr, code_table):
    rev_table = {v: k for k, v in code_table.items()}
    buffer, result = "", []
    for bit in bitstr:
        buffer += bit
        if buffer in rev_table:
            result.append(int(rev_table[buffer]))
            buffer = ""
    return result

### --- Сжатие и Восстановление с Хаффманом --- ###

def compress_image_huffman(image_path, q_y, q_c, value, namek, typek, block_size=8):
    image = Image.open(image_path)
    mode = image.mode
    if mode == 'L':  # Черно-белое изображение (один канал яркости)
        Y = np.array(image.convert('L')).astype(float)
        height, width = Y.shape

        blocks_Y = split_to_blocks(Y, block_size)

        all_data = []
        for block in blocks_Y:
            if block.shape != (block_size, block_size):

                continue  # Пропустить блок с неправильным размером

            dct_block = dct_2d(block)
            q_block = quantize(dct_block, q_y)
            zz = zigzag(q_block).astype(int)

            # Проверка длины данных после зигзага
            if len(zz) != block_size * block_size:

                continue  # Пропустить блок с неправильным размером

            all_data.extend(zz)


        huffman_table = build_huffman_tree(all_data)
        encoded_bits = huffman_encode(all_data, huffman_table)
        bitarr = bitarray.bitarray(encoded_bits)

        np.savez_compressed(f"comp/{namek}/dec_data/{typek}/compressed_{value}.npz",
                            mode="gray", Y=bitarr, huffman_table=huffman_table,
                            height=height, width=width, block_size=block_size)

    else:  # Цветное изображение
        image = image.convert('RGB')
        ycbcr = rgb_to_ycbcr(np.array(image).astype(float))
        Y, Cb, Cr = ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2]
        height, width = Y.shape

        all_data = []
        data_dict = {"Y": [], "Cb": [], "Cr": []}
        for name, channel, q in zip(["Y", "Cb", "Cr"], [Y, Cb, Cr], [q_y, q_c, q_c]):
            blocks = split_to_blocks(channel, block_size)
            for block in blocks:
                if block.shape != (block_size, block_size):

                    continue  # Пропустить блок с неправильным размером

                dct_block = dct_2d(block)
                q_block = quantize(dct_block, q)
                zz = zigzag(q_block).astype(int)
                all_data.extend(zz)
                data_dict[name].append(zz.tolist())

        # Отладочное сообщение, чтобы проверить длину всех данных


        huffman_table = build_huffman_tree(all_data)
        encoded_bits = huffman_encode(all_data, huffman_table)
        bitarr = bitarray.bitarray(encoded_bits)

        np.savez_compressed(f"comp/{namek}/dec_data/{typek}/compressed_{value}.npz",
                            mode="color", Y=bitarr, Cb=bitarr, Cr=bitarr,
                            huffman_table=huffman_table,
                            height=height, width=width, block_size=block_size)


def decompress_image_huffman(q_y, q_c, value, namek, typek, block_size=8):
    data = np.load(f"comp/{namek}/dec_data/{typek}/compressed_{value}.npz", allow_pickle=True)
    mode = data["mode"]
    huffman_table = data["huffman_table"].item()
    bitarr = bitarray.bitarray()
    bitarr.frombytes(data["Y"])

    decoded = huffman_decode(bitarr.to01(), huffman_table)

    height, width = data["height"], data["width"]

    if mode == "gray":  # Черно-белое изображение
        blocks = []
        block_count = 0

        # Проходим по всем декодированным данным и пытаемся восстановить блоки
        for i in range(0, len(decoded), block_size * block_size):
            zz = decoded[i:i + block_size * block_size]

            # Проверяем длину текущего блока
            if len(zz) != block_size * block_size:

                # Если блок слишком маленький, добавляем паддинг
                if len(zz) < block_size * block_size:
                    zz.extend([0] * (block_size * block_size - len(zz)))  # Паддинг нулями
                else:
                    continue  # Пропускаем блок, если он слишком большой

            block = inverse_zigzag(zz)
            block = idct_2d(dequantize(block, q_y))
            blocks.append(block)
            block_count += 1

        # Проверка, что количество блоков совпадает с размером изображения

        Y = merge_blocks(blocks, height, width, block_size)
        img = Image.fromarray(np.clip(np.round(Y), 0, 255).astype(np.uint8), mode='L')
        img.save(f"comp/{namek}/dec/{typek}/decompressed_image_{value}.png")

    else:  # Цветное изображение
        total_blocks = len(decoded) // (block_size * block_size * 3)
        blocks_Y, blocks_Cb, blocks_Cr = [], [], []
        for b in range(total_blocks * 3):
            zz = decoded[b * block_size * block_size:(b + 1) * block_size * block_size]

            # Проверка на корректный размер блока
            if len(zz) != block_size * block_size:

                continue  # Пропустить блок, если его размер некорректен

            block = inverse_zigzag(zz)
            q = q_y if b < total_blocks else q_c
            block = idct_2d(dequantize(block, q))
            if b < total_blocks:
                blocks_Y.append(block)
            elif b < 2 * total_blocks:
                blocks_Cb.append(block)
            else:
                blocks_Cr.append(block)

        Y = merge_blocks(blocks_Y, height, width, block_size)
        Cb = merge_blocks(blocks_Cb, height, width, block_size)
        Cr = merge_blocks(blocks_Cr, height, width, block_size)
        ycbcr = np.stack((Y, Cb, Cr), axis=2)
        rgb = ycbcr_to_rgb(ycbcr)
        Image.fromarray(rgb).save(f"comp/{namek}/dec/{typek}/decompressed_image_{value}.png")


### --- Стандартные матрицы квантования --- ###
q_y_std = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
])
q_c_std = np.array([
    [17,18,24,47,99,99,99,99],
    [18,21,26,66,99,99,99,99],
    [24,26,56,99,99,99,99,99],
    [47,66,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99]
])

### --- Функция масштабирования матрицы квантования --- ###
def scale_quant_matrix(matrix, quality):
    if quality <= 0:
        quality = 0.01

    if quality >= 100: quality = 100
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    scaled = np.floor((matrix * scale + 50) / 100)
    return np.clip(scaled, 1, 255)

def get_file_size(file_path):
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / 1024  # возвращаем размер в килобайтах
    else:
        return 0
def plot_file_size_graph(name, typek, quality_range):

            sizes = []
            values = []
            for value in quality_range:
                # Путь к файлу сжимаемых данных
                file_path = f"comp/{name}/dec_data/{typek}/compressed_{value}.npz"

                # Считываем размер файла
                size = get_file_size(file_path)
                sizes.append(size)
                values.append(value)

            # Создание нового графика для каждого типа и имени
            plt.figure(figsize=(8, 6))  # Устанавливаем размер графика
            plt.plot(values, sizes, label=f"{name} - {typek}")

            # Настройки графика
            plt.xlabel('Значение качества (quality)')
            plt.ylabel('Размер файла (KB)')
            plt.title(f'Зависимость размера файла от качества сжатия\n{name} - {typek}')
            plt.legend()
            plt.grid(True)

            # Показываем график
            graph_filename = f"{name}_{typek}_size_vs_quality.png"
            graph_path = os.path.join(save_folder, graph_filename)
            plt.savefig(graph_path)
            plt.close()

def main(image_path, quality,value,name,typek):
    q_y = scale_quant_matrix(q_y_std, quality)
    q_c = scale_quant_matrix(q_c_std, quality)
    compress_image_huffman(image_path, q_y, q_c,value,name,typek)
    decompress_image_huffman(q_y, q_c,value,name,typek)


if __name__ == "__main__":
    names=["Lenna","123"]
    types=["orig","gray","bw_no","bw_yes"]
    for name in names:
        for typek in types:
            for value in range(95, 101, 5):
                print(f"{name} {typek} {value},ready")
                main(f"prev/{name}/{typek}.png", value,value,name,typek)
            quality_range = range(0, 101, 5)
            save_folder = "graphs"
            plot_file_size_graph(name, typek, quality_range)