from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.train_pipeline import TrainPipeline

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def train_model():
    if request.method == "GET":
        return render_template("train.html")
    else:
        trainpipeline = TrainPipeline()
        r2score = trainpipeline.initiate_train_pipeline()
        return render_template(
            "train.html", msg=f"Training completed with {r2score}% accuracy"
        )


# age,sex,bmi,children,smoker,region,expenses
@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("predict.html")
    else:
        data = CustomData(
            age=int(request.form.get("age")),
            sex=str(request.form.get("sex")),
            bmi=float(request.form.get("bmi")),
            children=int(request.form.get("children")),
            smoker=str(request.form.get("smoker")),
            region=str(request.form.get("region")),
        )
        pred_df = data.get_data_as_data_frame()
        # print(pred_df)
        # print("Before Prediction")

        predict_pipeline = PredictPipeline()
        # print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        # print("after Prediction")
        return render_template("predict.html", results=round(results[0], 2), data=data)


if __name__ == "__main__":
    app.run(debug=True)
