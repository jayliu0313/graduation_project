from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session

db = SQLAlchemy()
# initialize our Flask application and the Keras model
app = Flask(__name__)
#Configure db
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:cvmle814@localhost:3306/flowerlive"
app.config['JSON_AS_ASCII'] = False   #chinese ascii decode and encode
db.init_app(app)

# 模型( model )定義
class flower_species(db.Model):
    __tablename__ = 'flower_species'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(45), unique=True)
    nickname = db.Column(db.String(200), nullable=False)
    scientific_name = db.Column(db.String(200), nullable=False)
    traits = db.Column(db.String(200), nullable=False)
    use = db.Column(db.String(200), nullable=False)
    link = db.Column(db.String(500))

    def __init__(self, id, name, nickname, scientific_name, traits, use, link):
        self.id = id
        self.name = name
        self.nickname = nickname
        self.scientific_name = scientific_name
        self.traits = traits
        self.use = use
        self.link = link
