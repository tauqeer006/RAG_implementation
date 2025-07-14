from flask import Flask
from flask import redirect
from flask import render_template
app = Flask(__name__)
@app.route("/")
def work():
    return render_template("work.html")
    