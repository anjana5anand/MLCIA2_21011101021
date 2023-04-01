#DBMS connection
import pymysql as ps 

db = ps.connect(
  host="localhost",
  user="root",
  password="Anjana@05",
  database="user_db"
)

cur= db.cursor()
cur.execute("SELECT * from users;")
data=cur.fetchall()
print(data)

#flask

from flask import Flask as f,request,render_template, url_for
import pandas as pd

def Hepatitisclassification(gmat,gpa,work_experience):
    import pandas as pd
    import matplotlib.pyplot as plt
    data = pd.read_csv(r"C:\Users\anjan\Downloads\HepatitisCdata.csv")
    data.head()
    data = data.dropna(axis=1)
    candidates = {'gmat': [780,750,690,710,680,730,690,720,740,690,610,690,710,680,770,610,580,650,540,590,620,600,550,550,570,670,660,580,650,660,640,620,660,660,680,650,670,580,590,690],
              'gpa': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,3.3,3.3,2.3,2.7,3.3,1.7,3.7],
              'work_experience': [3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5],
              'admitted': [1,1,0,1,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1]
              }
    data = pd.DataFrame(candidates,columns= ['gmat', 'gpa','work_experience','admitted'])
    print (data)
    X = data[['gmat', 'gpa','work_experience']]
    y = data['admitted']
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion='gini')
    tree.fit(X_train,y_train)
    y_pred=tree.predict(X_test[0:10])
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    new_candidates = {'gmat': [gmat], 'gpa': [gpa], 'work_experience': [work_experience]}
    df2 = pd.DataFrame(new_candidates,columns= ['gmat', 'gpa','work_experience'])
    y_pred=tree.predict(df2)
    print (df2) #test dataset
    print (y_pred) #predicted values
    return int(y_pred[0])



app=f(__name__)

@app.route("/")
def login():
    return render_template("login.html")

@app.route("/login",methods=["POST"])
def redirect():
    username=request.form["username"]
    passw = request.form["pswrd"]
    for i in data:
        print(i)
        if(username,passw) == i:
            url_for('input')
    return render_template("login.html")

@app.route("/input", methods=["POST"])
def input():
    return render_template("main.html")

@app.route("/receive",methods=["POST"])
def receive():
    if request.method == "POST" and request.form['submit'] == 'submit':
        gmat=request.form["gmat"]
        gpa = request.form["gpa"]
        work_experience = request.form["work_experience"]
        admitted = request.form["admitted"]
        y_pred=Hepatitisclassification(gpa,gmat,work_experience)
    if y_pred==1:
        predict="Positive"
    return render_template("index.html", predict=predict)

    
if __name__=='__main__':
    app.run(host='localhost',port=5000)


    
