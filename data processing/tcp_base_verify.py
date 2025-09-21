import numpy as np

# --- 请将这里替换为您自己的 tcp_base.npy 文件路径 ---
# 示例路径: '/home/ttt/BISE/RH20T/RH20T_cfg3/task_0001_user_0016_scene_0008_cfg_0003/transformed/tcp_base.npy'
file_path = '/home/ttt/BISE/RH20T_subset/task_0001/scene_1/human_pose.npy'

try:
    # 加载 .npy 文件
    # RH20T 的 .npy 文件通常包含Python对象（如字典），所以必须设置 allow_pickle=True
    data = np.load(file_path, allow_pickle=True)

    print(f"成功加载文件: {file_path}")
    print("-" * 50)

    # --- 1. 查看整体数据类型和结构 ---
    print("1. 整体数据分析:")
    print(f"   - 加载后数据的类型: {type(data)}")

    # 根据RH20T的结构，它通常是一个被包裹在0维数组里的字典
    if isinstance(data, np.ndarray) and data.shape == ():
        print("   - 数据是一个被Numpy数组包裹的Python对象。")
        data = data.item() # 使用 .item() 将其从数组中取出
        print(f"   - 取出后，对象类型为: {type(data)}")

    print("-" * 50)

    # --- 2. 深入分析数据内容 (假设它是一个字典) ---
    print("2. 数据内容详解:")
    if isinstance(data, dict):
        print(f"   - 这是一个字典，包含 {len(data)} 个键。")
        # 字典的键通常是相机序列号
        keys = list(data.keys())
        print(f"   - 字典的键 (通常是相机序列号): {keys}")

        # 我们只查看第一个键对应的内容作为示例
        first_key = keys[1]
        print(f"\n   --- 以键 '{first_key}' 为例进行分析 ---")

        trajectory_data = data[first_key]
        print(f"   - 键 '{first_key}' 对应值的类型: {type(trajectory_data)}")

        if isinstance(trajectory_data, list) and len(trajectory_data) > 0:
            print(f"   - 这是一个列表，包含 {len(trajectory_data)} 个时间点的记录。")
            
            # 查看第一个时间点的数据结构
            first_timestamp_data = trajectory_data[0]
            print("\n   - 列表内第一个元素 (第一个时间点) 的内容:")
            print(f"     {first_timestamp_data}")
            print(f"     类型: {type(first_timestamp_data)}")
            
            if isinstance(first_timestamp_data, dict):
                print("\n   - 对第一个时间点的数据进行格式解析:")
                timestamp = first_timestamp_data.get('timestamp', '未找到')
                tcp_pose = first_timestamp_data.get('tcp', '未找到')
                
                print(f"     - 时间戳 (timestamp): {timestamp}")
                print(f"     - TCP位姿 (tcp): {tcp_pose}")
                
                if isinstance(tcp_pose, np.ndarray):
                    print(f"     - TCP位姿的数据类型: {type(tcp_pose)}")
                    print(f"     - TCP位姿的形状 (shape): {tcp_pose.shape}")
                    print("       (这应该是 (7,)，代表 xyz + 四元数quat)")
        else:
            print("   - 该键对应的值不是预期格式，请直接查看其内容:")
            print(trajectory_data)
    else:
        print("   - 数据不是预期的字典格式，请直接查看其shape:")
        print(data.shape)


except FileNotFoundError:
    print(f"错误: 文件未找到，请检查路径是否正确: {file_path}")
except Exception as e:
    print(f"发生了一个错误: {e}")