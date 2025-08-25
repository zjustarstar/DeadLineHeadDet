import cv2
import os
import numpy as np
import torch
import time
import fitz  # PyMuPDF
from PIL import Image
from datetime import datetime
import classifier_infer as infer
from skimage import morphology

def get_Ap(binary_img, y, x):
    if binary_img[y, x] == 0:
        return 0  # 背景点直接返回0

    # 取八邻域像素
    P2 = binary_img[y - 1, x]
    P3 = binary_img[y - 1, x + 1]
    P4 = binary_img[y, x + 1]
    P5 = binary_img[y + 1, x + 1]
    P6 = binary_img[y + 1, x]
    P7 = binary_img[y + 1, x - 1]
    P8 = binary_img[y, x - 1]
    P9 = binary_img[y - 1, x - 1]

    neighbors = [P2, P3, P4, P5, P6, P7, P8, P9, P2]  # 闭合环用于过渡计算
    transitions = 0
    for i in range(8):
        if neighbors[i] == 0 and neighbors[i + 1] == 1:
            transitions += 1
    return transitions


def find_endpoints(gray_image, window_size=40):
    # 二值化图像，前景为白（原图前景为黑则使用 THRESH_BINARY_INV）
    thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)[1]

    # cv2.imwrite("output\\thresh.png", thresh)

    # # 使用中值滤波去噪（3x3）
    # thresh = cv2.medianBlur(thresh, 3)

    # 使用OpenCV的细化算法获得骨架
    #thinned = cv2.ximgproc.thinning(thresh)

    thresh[thresh == 255] = 1
    thinned = morphology.skeletonize(thresh)  # 骨架提取
    # thinned[thinned == 1] = 255
    # # cv2.imwrite("output\\s.png", thinned)
    #
    # # 将骨架图转为二值（0/1）格式
    thinned_bin = (thinned == 1).astype(np.uint8)

    # 定义一个卷积核，用于检测八邻域之和
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)

    # 卷积操作，寻找中心点 + 八邻域 == 11 的点（即中心是1，邻域有1个值为1）
    conv = cv2.filter2D(thinned_bin, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    # 找到候选点（值为11的点）
    check_points = np.where(conv == 11)

    # 去除图像边界一定宽度内的点，避免越界处理
    mask = (check_points[0] > window_size) & (check_points[0] < gray_image.shape[0] - window_size) & \
           (check_points[1] > window_size) & (check_points[1] < gray_image.shape[1] - window_size)
    check_points = (check_points[0][mask], check_points[1][mask])

    return thinned_bin, check_points


def detect(gray_image, board=40, thre_deadline_length=38):
    # 骨架图
    thinned_bin, check_points = find_endpoints(gray_image, board)

    # temp = thinned_bin.astype(np.uint8) * 255
    # cv2.imwrite("output\\sketch.png", temp)

    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)

    final_points = []
    
    # 遍历所有候选点
    for cy, cx in zip(*check_points):
        half_width = board
        cropped = False  # 标记是否重新裁剪过区域
        
        while True:
            # 提取以候选点为中心的区域
            top = max(cy - half_width, 0)
            bottom = min(cy + half_width, thinned_bin.shape[0] - 1)
            left = max(cx - half_width, 0)
            right = min(cx + half_width, thinned_bin.shape[1] - 1)
            area = thinned_bin[top:bottom, left:right]
            center_y = cy - top
            center_x = cx - left

            # 获取与中心连通的前景区域
            labels = cv2.connectedComponents(area, connectivity=8)[1]
            mask = (labels == labels[(center_y, center_x)]).astype(np.uint8)
            
            # 检测交叉点（通过局部卷积）
            conv_local = cv2.filter2D(area, -1, kernel, borderType=cv2.BORDER_CONSTANT)
            cross_points = np.where(conv_local >= 13)

            filtered_cross_points = []
            for y, x in zip(cross_points[0], cross_points[1]):
                if y == 0 or y == area.shape[0] - 1 or x == 0 or x == area.shape[1] - 1:
                    continue  # 去除边界
                Ap = get_Ap(area, y, x)
                if Ap >= 3:
                    filtered_cross_points.append((y, x))

            # 考虑2*2的交叉点
            conv_local = cv2.filter2D(mask, -1, np.array([[1, 1], [1, 1]]), borderType=cv2.BORDER_CONSTANT)
            cross_points = np.where(conv_local == 4)
            for y, x in zip(cross_points[0], cross_points[1]):
                filtered_cross_points.append((y, x))

            if len(filtered_cross_points) == 0:
                break  # 无交叉点，跳过该点
            
            # 排序交叉点到中心的曼哈顿距离
            distances = [np.abs(y - center_y) + np.abs(x - center_x) for y, x in filtered_cross_points]
            sorted_indices = np.argsort(distances)
            
            # 多交叉点时，重新裁剪区域，使次近点落在边缘
            if len(filtered_cross_points) > 1 and not cropped:
                h1 = max(np.abs(filtered_cross_points[sorted_indices[0]][1] - center_x),
                         np.abs(filtered_cross_points[sorted_indices[0]][0] - center_y))
                h2 = max(np.abs(filtered_cross_points[sorted_indices[1]][1] - center_x),
                         np.abs(filtered_cross_points[sorted_indices[1]][0] - center_y))
                half_width = max(h1 + 2, h2 + 1)
                cropped = True
                continue
            
            # 提取边缘点（与 mask 相交的四边）
            edge_mask = np.zeros_like(mask, dtype=np.uint8)
            edge_mask[0, :] = edge_mask[-1, :] = edge_mask[:, 0] = edge_mask[:, -1] = 1
            edge_points = edge_mask * mask
            
            edge_points_num, edge_points_labels = cv2.connectedComponents(edge_points, connectivity=8)
            edge_point = []
            for i in range(1, edge_points_num):
                y, x = np.where(edge_points_labels == i)
                if len(y) == 1:
                    edge_point.append((y[0], x[0]))
                elif len(y) > 1:
                    p1 = (y[0], x[0])
                    p2 = (y[-1], x[-1])
                    dist1 = np.abs(p1[0] - half_width) + np.abs(p1[1] - half_width)
                    dist2 = np.abs(p2[0] - half_width) + np.abs(p2[1] - half_width)
                    if dist1 > dist2:
                        edge_point.append(p2)
                    else:
                        edge_point.append(p1)
            
            if len(edge_point) < 2:
                break
            
            # 若边缘过多，仅保留最远的一对
            if len(edge_point) > 2:
                max_dist = -1
                p1, p2 = None, None
                for i in range(len(edge_point)):
                    for j in range(i + 1, len(edge_point)):
                        d = np.linalg.norm(np.array(edge_point[i]) - np.array(edge_point[j]))
                        if d > max_dist:
                            max_dist = d
                            p1, p2 = edge_point[i], edge_point[j]
                edge_point = [p1, p2]
            
            # 判断线头类型 ,判断交叉点是否方向稳定
            cross_point = filtered_cross_points[sorted_indices[0]]
            center_point = (center_y, center_x)
            
            v0 = np.array(center_point) - np.array(cross_point)
            v1 = np.array(edge_point[0]) - np.array(cross_point)
            v2 = np.array(edge_point[1]) - np.array(cross_point)

            # v0不能太长;
            v0_length = (center_point[0]-cross_point[0])*(center_point[0]-cross_point[0]) + \
            (center_point[1] - cross_point[1]) * (center_point[1] - cross_point[1])
            if v0_length > thre_deadline_length * thre_deadline_length:
                break
            
            v0_norm = v0 / (np.linalg.norm(v0) + 1e-8)
            v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
            v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
            
            cos01 = np.dot(v0_norm, v1_norm)
            cos02 = np.dot(v0_norm, v2_norm)
            cos12 = np.dot(v1_norm, v2_norm)
            
            # 判断向量v0是否在v1和v2之间（基于夹角余弦）
            if cos01 + cos02 > 0:
                final_points.append((cy, cx))
            else:
                # 如果夹角小于150度，也认为是合法分支
                if cos12 < np.cos(np.deg2rad(120)):
                    final_points.append((cy, cx))

            break

    return final_points


def save_endpoint_regions(image, data_path, coordinates, region_size=100):
    """
    保存线头区域到rect_region目录，用于后期训练

    Args:
        image: 原始图像
        coordinates: 线头坐标列表 [(x1, y1), (x2, y2), ...]
        region_size: 区域大小，如果为None则从配置文件读取

    Returns:
        保存的区域路径列表
    """
    # 创建保存区域的目录
    region_dir = ".\\rect_regions"
    os.makedirs(region_dir, exist_ok=True)

    saved_regions = []
    height, width = image.shape[:2]

    for i, (y, x) in enumerate(coordinates):
        # 计算区域的左上角和右下角坐标
        half_size = region_size // 2
        left = max(0, x - half_size)
        top = max(0, y - half_size)
        right = min(width, x + half_size)
        bottom = min(height, y + half_size)

        # 如果区域太小，则跳过
        if right - left < 10 or bottom - top < 10:
            print(f"警告: 区域 {i} 太小，跳过: ({left}, {top}, {right}, {bottom})")
            continue

        # 提取区域
        region = image[top:bottom, left:right]

        # 如果区域不是正方形，则进行填充
        if right - left != bottom - top:
            # 确定目标大小（取最大边长）
            target_size = max(right - left, bottom - top)
            # 创建白色背景的正方形图像
            square_region = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
            # 计算放置位置（居中）
            offset_x = (target_size - (right - left)) // 2
            offset_y = (target_size - (bottom - top)) // 2
            # 将原始区域放入正方形图像
            square_region[offset_y:offset_y + (bottom - top), offset_x:offset_x + (right - left)] = region
            region = square_region

        # 调整大小为固定尺寸
        region = cv2.resize(region, (region_size, region_size))

        # 保存区域
        filename = os.path.basename(data_path)
        name, extension = os.path.splitext(filename)
        timestamp = datetime.now().strftime("%m%d_%H%M%S_%f")[:-3]  # 精确到毫秒
        region_path = os.path.join(region_dir, f"{name}_{timestamp}.png")
        cv2.imwrite(region_path, region)

    print(f"已保存 {len(coordinates)} 个线头区域到 {region_dir} 目录")
    return saved_regions


def filter_dense_clusters(points, eps=170, min_samples=4, max_cluster_size=3):
    """
    过滤过于密集的点群
    
    Args:
        points: 检测到的线头点列表，每个元素为(y, x)
        eps: DBSCAN聚类的邻域半径
        min_samples: DBSCAN聚类的最小样本数
        max_cluster_size: 允许的最大聚类大小，超过此值的聚类将被移除
    
    Returns:
        list: 过滤后的线头点列表
    """
    if len(points) < 30:
        return points  # 点数太少，不需要过滤
    
    try:
        # 将点列表转换为numpy数组
        points_array = np.array(points)
        
        # 使用DBSCAN聚类算法识别密集点群
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_array)
        
        # 获取每个点的聚类标签
        labels = clustering.labels_
        
        # 统计每个聚类的点数
        from collections import Counter
        cluster_counts = Counter(labels[labels != -1])  # 排除噪声点（标签为-1）
        
        # 找出过于密集的聚类（点数超过阈值）
        dense_clusters = [cluster_id for cluster_id, count in cluster_counts.items() 
                         if count > max_cluster_size]
        
        # 如果存在过于密集的聚类，移除这些聚类中的所有点
        if dense_clusters:
            print(f"发现{len(dense_clusters)}个过于密集的点群，共包含{sum([cluster_counts[c] for c in dense_clusters])}个点")
            mask = np.ones(len(points), dtype=bool)
            for cluster_id in dense_clusters:
                mask = mask & (labels != cluster_id)
            
            # 返回过滤后的点
            filtered_points = [points[i] for i in range(len(points)) if mask[i]]
            print(f"过滤前点数: {len(points)}, 过滤后点数: {len(filtered_points)}")
            return filtered_points
        else:
            return points  # 没有过于密集的聚类，返回原始点列表
    
    except Exception as e:
        print(f"过滤密集点群时出错: {e}")
        return points  # 出错时返回原始点列表


def filter_by_classifier(model, device, image, coordinates):
    height, width = image.shape[:2]
    region_size = 100
    final_point = []

    for i, (y, x) in enumerate(coordinates):
        # 计算区域的左上角和右下角坐标
        half_size = region_size // 2
        left = max(0, x - half_size)
        top = max(0, y - half_size)
        right = min(width, x + half_size)
        bottom = min(height, y + half_size)

        # 如果区域太小，则跳过
        if right - left < 10 or bottom - top < 10:
            print(f"警告: 区域 {i} 太小，跳过: ({left}, {top}, {right}, {bottom})")
            continue

        # 提取区域
        region = image[top:bottom, left:right]

        # 如果区域不是正方形，则进行填充
        if right - left != bottom - top:
            # 确定目标大小（取最大边长）
            target_size = max(right - left, bottom - top)
            # 创建白色背景的正方形图像
            square_region = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
            # 计算放置位置（居中）
            offset_x = (target_size - (right - left)) // 2
            offset_y = (target_size - (bottom - top)) // 2
            # 将原始区域放入正方形图像
            square_region[offset_y:offset_y + (bottom - top), offset_x:offset_x + (right - left)] = region
            region = square_region

        # 调整大小为固定尺寸
        region = cv2.resize(region, (region_size, region_size))

        # 转为PIL格式:
        cv_img_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        # 转为 PIL Image
        pil_img = Image.fromarray(cv_img_rgb)

        # 进行推理;
        trans = infer.create_data_transforms()
        class_id, prob = infer.infer_main(model, trans, pil_img, device=device)
        if not (class_id <= 4 and prob>0.7):
            final_point.append(coordinates[i])

    return final_point


def preprocess_image(image):
    """
    预处理图像，将其转换为灰度图
    
    Args:
        image_path: 图像文件路径
    
    Returns:
        tuple: (原始图像, 灰度图像)
    """
    
    # 将图像转换为灰度图
    if len(image.shape) == 3:
        # 如果是带透明通道的RGBA图像
        if image.shape[2] == 4:
            b, g, r, a = cv2.split(image)
            image_rgb = cv2.merge([b, g, r])
            image_rgb[a == 0] = [255, 255, 255]
            gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
        # 如果是RGB图像
        elif image.shape[2] == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # 如果已经是灰度图
        gray_image = image.copy()

    return gray_image


def draw_branch_points(image, branch_points, color=(0, 0, 255), radius=60, thickness=8):
    """
    在图像上绘制分支点
    
    Args:
        image: 输入图像
        branch_points: 分支点列表，每个元素为(y, x)
        color: 绘制颜色，默认绿色
        radius: 圆圈半径
        thickness: 线条粗细
    
    Returns:
        numpy.ndarray: 绘制后的图像
    """
    result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # 绘制分支点
    for cy, cx in branch_points:
        cv2.circle(result_image, (cx, cy), radius, color, thickness)
    
    return result_image


def is_pdf_by_extension(file_path):
    _, ext = os.path.splitext(file_path)
    return ext.lower() == '.pdf'


def change_extension_to_png(file_path):
    # 分离文件名和扩展名
    base, _ = os.path.splitext(file_path)
    # 构建新的文件名
    new_file_path = base + '.png'
    return new_file_path


def deal_single(data_path, model, device, output_dir=""):
    try:
        if is_pdf_by_extension(data_path):
            img, _ = pdf_to_image(data_path, 0)
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            # 图像预处理
            img = cv2.imread(data_path, cv2.IMREAD_UNCHANGED)
            gray_image = preprocess_image(img)
        print(f"灰度图像尺寸: {gray_image.shape}")

        # cv2.imwrite("image.png", img)

        # 检测分支点
        branch_points = detect(gray_image)
        print(f"检测到的分支点数量: {len(branch_points)}")
        # 保存检测到的roi区域，用于后续训练;
        # save_endpoint_regions(img, data_path, branch_points)

        # 过滤过于集中的点（可能是误检测）
        filtered_points = filter_dense_clusters(branch_points)
        print(f"密度过滤后的分支点数量: {len(filtered_points)}")

        # 如果点不多就不再过滤;
        final_points = filtered_points
        if len(filtered_points) > 25:
            final_points = filter_by_classifier(model, device, img, filtered_points)
        print(f"cnn过滤后的分支点数量: {len(final_points)}")

        # 根据output_dir来判断是会否保存一些测试图
        output_path=""
        if output_dir != "":
            # 绘制分支点
            print("正在绘制结果...")

            # 保存结果图像
            if is_pdf_by_extension(data_path):
                data_path = change_extension_to_png(data_path)
            filename = os.path.basename(data_path)
            name, extension = os.path.splitext(filename)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"目录 {output_dir} 不存在，已创建")

            # 最初始的结果
            result_image_before = draw_branch_points(gray_image, branch_points)
            output_path = os.path.join(output_dir, f"{name}_aa{extension}")
            cv2.imwrite(output_path, result_image_before)

            # dense过滤的结果
            result_image_dense = draw_branch_points(gray_image, filtered_points)
            output_path = os.path.join(output_dir, f"{name}_after_dense{extension}")
            cv2.imwrite(output_path, result_image_dense)

            #最终的结果
            result_image_classifier = draw_branch_points(gray_image, final_points)
            output_path = os.path.join(output_dir, f"{name}_final{extension}")
            cv2.imwrite(output_path, result_image_classifier)
            print(f"结果已保存到: {output_path}")

        return final_points

    except FileNotFoundError:
        print(f"错误: 找不到文件 '{data_path}'")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")


def deal_batch(dir_path, model, device, output_dir):
    # 获取数据文件列表
    if not os.path.exists(dir_path):
        print(f"错误: 数据目录不存在: {dir_path}")
        return

    data_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf'))]

    if not data_files:
        print(f"警告: 数据目录中没有找到图像文件: {dir_path}")
        return

    # 处理每个数据文件
    i = 0
    pts = []
    for image_path in data_files:
        print(f"\n处理图像 {i+1}/{len(data_files)}: {image_path}")
        pt = deal_single(image_path, model, device, output_dir)
        pts.append(len(pt))
        i = i + 1

    print(f"minPts={min(pts)}, maxPts={max(pts)}, avgPts={sum(pts)/len(data_files)}")

    return len(data_files)


def pdf_to_image(pdf_path, page_num=0):
    """
    将PDF文件的指定页面转换为图像

    Args:
        pdf_path: PDF文件路径
        page_num: 页码（从0开始）

    Returns:
        PIL图像对象和页面尺寸信息
    """
    pdf_dpi = 300  # PDF转图像的DPI

    # 打开PDF文件
    doc = fitz.open(pdf_path)
    if page_num >= len(doc):
        raise ValueError(f"PDF只有{len(doc)}页，无法访问第{page_num + 1}页")

    # 获取指定页面
    page = doc[page_num]

    # 获取页面尺寸（PDF坐标系）
    pdf_width = page.rect.width
    pdf_height = page.rect.height

    # 渲染页面为图像
    pix = page.get_pixmap(matrix=fitz.Matrix(pdf_dpi / 72, pdf_dpi / 72))

    # 转换为PIL图像
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # 转换为OpenCV格式（BGR）
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    if doc:
        doc.close()

    # 返回图像和页面尺寸信息
    return img_cv, (pdf_width, pdf_height, pix.width, pix.height)


if __name__ == "__main__":

    # 加载分类器模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型,注意只加载一次
    model_path = ".\\models\\classifier.pth"
    model = infer.load_model(model_path, device)

    start_time = time.time()
    # 单个文件
    output_dir = "output"
    data_path = "data\\pdf\\t2.png"
    file_size = 1
    # 如果不需要输出,则只需要输入前三个参数，或者将output_dir设为""
    #points = deal_single(data_path, model, device, output_dir)
    # print(f'最终点数:{len(points)}')

    # 批处理
    dir_path = "data\\pdf"
    file_size = deal_batch(dir_path, model, device, output_dir)
    # 记录结束时间
    end_time = time.time()

    # 计算运行时长（秒）
    duration = (end_time - start_time) // file_size

    # 转换为更友好的格式（可选）
    minutes, seconds = divmod(duration, 60)
    hours, minutes = divmod(minutes, 60)

    # 输出结果
    print(f"每张图运行时长: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")