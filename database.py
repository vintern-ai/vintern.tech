from deta import Deta
from dotenv import load_dotenv
import os
#from dotenv import load_dotenv  # pip install python-dotenv

#DETA_KEY="a0rsyxu6_5ZWz41gJtxwMCS4pBY4qTa4oihamzyp4"

DETA_KEY1="a0l8hnjf_BiRSRUKyJiG9Zn8tc1eCMSf9TayAGVhw"
load_dotenv(".env")
DETA_KEY=os.getenv("DETA_KEY")
deta1=Deta(DETA_KEY1)
deta=Deta(DETA_KEY)
db=deta.Base("VINTERNS_")
db1=deta1.Base("VI")
def insertuser(email, name, password):
    return db.put({"key":email, "name":name, "password": password})



def fetch_all__users():
    res=db.fetch()
    return res.items

#print(fetch_all__users())

abc="jsadjb"

insertuser("admin@123", "admin", "abc163")

#db.put({"key":"admin@123","vi":"hi"})





