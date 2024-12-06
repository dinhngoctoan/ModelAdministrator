
"""
https://fastapi.tiangolo.com/advanced/custom-response/#html-response
"""
#from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from model.RandomForest import RanForestModel
from model.RNN import RNNModel
from model.LSTM import LSTMModel
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from io import BytesIO
import base64
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns
app = FastAPI()
app.mount("/assets", StaticFiles(directory="templates/assets"), name="assets")
templates = Jinja2Templates(directory="templates/")
data = pd.read_csv("templates/assets/Data/creditcard.csv")
data1 = data.sample(frac=1)
# amount of fraud classes 400 rows.
fraud_df = data1.loc[data1['Class'] == 1][:400]
non_fraud_df = data1.loc[data1['Class'] == 0][:400]
normal_distributed_df = pd.concat([fraud_df, non_fraud_df])
df = normal_distributed_df.sample(frac=1, random_state=42)
y = df['Class']
X = df.drop(['Class','Time'],axis=1)
'''version_data = [
    {"version": 1, "detail": "Chi tiết phiên bản 1"},
    {"version": 2, "detail": "Chi tiết phiên bản 2"},
    {"version": 3, "detail": "Chi tiết phiên bản 3"},
]'''
RanModel = RanForestModel()
RNN_Model = RNNModel(X,y)
LSTM_Model = LSTMModel(X,y)
models = [
    {"name": "RandomForest", "model":RanModel},
    {"name": "RNN", "model":RNN_Model},
    {"name": "LSTM", "model": LSTM_Model}
]

def read_data(datafile):
    df = pd.read_csv(f"templates/assets/Data/{datafile}")
    return df

def pie_chart(df):
    counts = df['Is Fraud?'].value_counts()
    fig = plt.figure()#figsize=(4,4))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    # Chuyển đổi hình ảnh thành Base64
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return img_base64
def bar_graph(df):
    df['Amount'] = df['Amount'].str.replace("$",'').astype(float)
    fraud = df.loc[df['Is Fraud?'] == 'Yes']
    fraudunder100 = fraud.loc[fraud['Amount'] < 100.0]
    fraudover100 = fraud.loc[fraud['Amount']>100.0]
    data = {'Under $100':fraudunder100.shape[0],'Over $100':fraudover100.shape[0]}
    courses = list(data.keys())
    values = list(data.values())
    fig = plt.figure()#figsize = (10, 5))
    plt.bar(courses, values, color ='red', width = 0.4)
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    # Chuyển đổi hình ảnh thành Base64
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return img_base64
def time_graph(df):
    df = df[df["Time"]<=86400]
    threshold = 100
    bins = [0, 21600, 43200, 64800, 86400]
    labels = ["Buổi đêm", "Buổi sáng", "Buổi trưa chiều", "Buổi tối"]
    df["time_group"] = pd.cut(df["Time"], bins=bins, labels=labels, right=False)
    result = df.groupby("time_group")["Amount"].apply(
        lambda x: pd.Series({
            "below_threshold": (x < threshold).sum(),
            "above_threshold": (x >= threshold).sum()
        })
        ).unstack()
    fig1, ax = plt.subplots()

    # Vị trí và chiều rộng cột
    x = np.arange(len(labels))
    width = 0.35

    # Vẽ từng nhóm cột
    bars1 = ax.bar(x - width/2, result["below_threshold"], width, label="Dưới 100$", color="skyblue")
    bars2 = ax.bar(x + width/2, result["above_threshold"], width, label="Trên 100$", color="orange")
    ax.bar_label(bars1, fmt='%d', padding=3)
    ax.bar_label(bars2, fmt='%d', padding=3)

    # Cài đặt nhãn và tiêu đề
    ax.set_xlabel("Buổi")
    ax.set_ylabel("Số lượng trong ngày")

    #ax.set_title("Số lượng giao dịch tại mỗi khung thời gian")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    # Chuyển đổi hình ảnh thành Base64
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")


    fraud_sum = (
    df[df["Class"] == 1]  # Chỉ lấy các giao dịch gian lận
    .groupby("time_group")["Amount"]
    .sum()
)

# Đảm bảo các khung giờ có giá trị 0 nếu không có giao dịch
    fraud_sum = fraud_sum.reindex(labels, fill_value=0)
    plt.figure(figsize=(8, 6))
    plt.plot(fraud_sum.index, fraud_sum.values, marker='o', color='red', label="Fraud Transactions")

# Thêm nhãn và tiêu đề
    plt.title("Tổng số tiền giao dịch gian lận theo khung giờ", fontsize=14)
    plt.xlabel("Khung giờ", fontsize=12)
    plt.ylabel("Tổng số tiền (USD)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()

    buf2 = BytesIO()
    plt.savefig(buf2, format="png")
    plt.close()
    buf2.seek(0)
    img_base64_line = base64.b64encode(buf2.read()).decode("utf-8")
    return img_base64, img_base64_line
def plot_confusion_matrix(y_test, y_prediction):


    # Tính confusion matrix và chuyển sang dạng phần trăm
    cm = confusion_matrix(y_test, y_prediction)
    percentages_matrix = (cm / np.sum(cm, axis=None, keepdims=True)) * 100

    # Tạo figure và axes
    ax = plt.subplot()  # Điều chỉnh kích thước biểu đồ
    sns.heatmap(
        percentages_matrix, 
        annot=True, 
        vmin=0, 
        vmax=100, 
        fmt='.2f', 
        cmap="crest", 
        annot_kws={"fontsize": 14, "fontweight": "bold"},
        ax=ax
    )

    # Cài đặt nhãn và tiêu đề
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(["Fraud", "Not Fraud"])
    ax.yaxis.set_ticklabels(["Fraud", "Not Fraud"])
    
    # Thêm dấu % vào các giá trị hiển thị
    for t in ax.texts:
        t.set_text(t.get_text() + "%")

    # Áp dụng bố cục và lưu vào buffer
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')  # Lưu trực tiếp vào buffer
    buf.seek(0)
    plt.close()  # Đóng figure sau khi xử lý xong

    # Chuyển đổi hình ảnh thành Base64
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return img_base64
@app.get("/", response_class=HTMLResponse)
async def read_items():
    return """
    <html>
        <head>
            <title>Simple HTML app</title>
        </head>
        <body>
            <h1>Navigate to <a href="http://localhost:8000/Dashboard">Dashboard</a></h1>
        </body>
    </html>
    """


@app.get("/Dashboard")
async def DashboardUI(request: Request):
    df = read_data("transaction.csv")
    pie_chart_img = pie_chart(df)
    bar_graph_img = bar_graph(df)
    df = read_data("creditcard.csv")
    time_bar_graph_img, time_line_graph_img= time_graph(df)
    return templates.TemplateResponse(
        "Dashboard.html", context={"request": request, "pie_chart":pie_chart_img, "bar_graph":bar_graph_img,"time_bar_graph":time_bar_graph_img,"time_line_graph":time_line_graph_img}
    )



@app.get("/model")
async def modelUI(request: Request):
    return templates.TemplateResponse(
        "model.html", context={"request": request, "models":models}
    )

@app.post("/model/{model_id}")
async def trainModel(request: Request,model_id: int):
    data1 = data.sample(frac=1)
# amount of fraud classes 400 rows.
    fraud_df = data1.loc[data1['Class'] == 1][:400]
    non_fraud_df = data1.loc[data1['Class'] == 0][:400]
    normal_distributed_df = pd.concat([fraud_df, non_fraud_df])
    df = normal_distributed_df.sample(frac=1, random_state=42)
    y = df['Class']
    X = df.drop(['Class','Time'],axis=1)
    if(model_id == 1):
        accuracy = models[model_id-1]['model'].train(X,y)
    else:
        accuracy = models[model_id-1]['model'].train(X,y)
    return {"accuracy":accuracy}

@app.post("/model/setVersion/{model_id}/{id}")
async def setModel(request:Request, model_id:int , id: int):
    print(model_id)
    models[model_id]['model'].setModel(id)
    return {"status":"success"}

@app.get("/output")
async def outputUI(request:Request):
    return templates.TemplateResponse(
        "output.html", context={"request": request,"models":models}
    )
@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...),model_id: str = Form(...)):
    if file.content_type != "text/csv":
        return templates.TemplateResponse(
            "output.html",
            {"request": request, "error": "File không hợp lệ. Vui lòng tải lên file CSV."},
        )
    if not model_id:
        return {"error": "model_id không được bỏ trống"}
    model_id = int(model_id)
    y_predict = None
    try:
        # Đọc file CSV
        df = pd.read_csv(file.file)
        data = df.drop(['Time','Class'],axis = 1)
        y_test = df['Class']
        # Lọc cột mặc định
        if(model_id==0):
            print("Predicting with model 0")
            y_predict = models[model_id]['model'].predict(data)
        else:
            print(f"Predicting with model {model_id}")
            data = data.values.reshape((data.shape[0], 1, data.shape[1]))
            print(model_id)
            y_pre = models[model_id]['model'].predict(data)
            y_predict_forecast = [x[0] for x in y_pre]
            y_predict = (y_pre > 0.5).astype("int32")
            y_predict = [x[0] for x in y_predict]
        cfs_matrix  = plot_confusion_matrix(y_test,y_predict)
        classifi_report = classification_report(y_test,y_predict)
        print("Success")
        # Trả về dữ liệu cùng giao diện
        return templates.TemplateResponse(
            "output.html",
            context = {"request": request, "columns": ['Readlity','Predict'], "reality":y_test.tolist(),
                       "predict": y_predict if model_id==0 else [round(val, 2) for val in y_predict_forecast], "cfs_matrix":cfs_matrix, "classification_report":classifi_report},#
        )
    except Exception as e:
        print("Error occurred during processing:", e)
        return templates.TemplateResponse(
            "output.html",
            {"request": request, "error": f"Không thể xử lý file CSV: {str(e)}"},
        )
'''
@app.post("/model/RandomForest")
async def trainModel(request: Request):
    data1 = data.sample(frac=1)
    # amount of fraud classes 492 rows.
    fraud_df = data1.loc[data1['Class'] == 1]
    non_fraud_df = data1.loc[data1['Class'] == 0][:492]
    normal_distributed_df = pd.concat([fraud_df, non_fraud_df])
    df = normal_distributed_df.sample(frac=1, random_state=42)
    y = df['Class']
    X = df.drop(['Class','Time'],axis=1)
    model = RanForestModel()
    accuracy = model.train(X,y)
    return {"accuracy":accuracy}

@app.post("/model/LSTM")
async def trainModel(request: Request):
    data1 = data.sample(frac=1)
    # amount of fraud classes 492 rows.
    fraud_df = data1.loc[data1['Class'] == 1]
    non_fraud_df = data1.loc[data1['Class'] == 0][:492]
    normal_distributed_df = pd.concat([fraud_df, non_fraud_df])
    df = normal_distributed_df.sample(frac=1, random_state=42)
    y = df['Class']
    X = df.drop(['Class','Time'],axis=1)
    model = LSTMModel(X,y)  
    accuracy = model.train()
    return {"accuracy":accuracy}

@app.post("/model/RNN")
async def trainModel(request: Request):
    data1 = data.sample(frac=1)
    # amount of fraud classes 492 rows.
    fraud_df = data1.loc[data1['Class'] == 1]
    non_fraud_df = data1.loc[data1['Class'] == 0][:492]
    normal_distributed_df = pd.concat([fraud_df, non_fraud_df])
    df = normal_distributed_df.sample(frac=1, random_state=42)
    y = df['Class']
    X = df.drop(['Class','Time'],axis=1)
    model = RNNModel(X,y)  
    accuracy = model.train()
    return {"accuracy":accuracy}
'''