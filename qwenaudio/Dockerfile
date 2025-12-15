# 使用PyTorch官方镜像作为基础（已包含CUDA支持）
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# 设置工作目录
WORKDIR /app

# 先复制依赖文件以利用Docker缓存层
COPY requirements.txt .

# 安装Python依赖（使用清华镜像加速）
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini && \
    pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制整个项目到容器
COPY . .

# 声明容器运行时需要映射的端口（根据实际需要修改）
EXPOSE 8080

# 设置容器启动命令
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "scripts/infer_service.py"]