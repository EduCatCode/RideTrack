# 使用基本的Miniconda映像作為基礎
FROM continuumio/miniconda3

# 設定工作目錄
WORKDIR /workspace

# 將專案文件複製到工作目錄
COPY . .

# 創建並激活conda環境
RUN conda create --name RideTrack python=3.7
SHELL ["conda", "run", "-n", "RideTrack", "/bin/bash", "-c"]

# 安裝所需的套件
RUN pip install -r requirements.txt

CMD ["/bin/bash"]