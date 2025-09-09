from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime
from dateutil.relativedelta import relativedelta
import io
import csv
import pandas as pd

# -------------------------
# Configuration
# -------------------------
MONGO_URI = "mongodb://localhost:27017"   # change if using Atlas or remote Mongo
DB_NAME = "expense_manager"

app = Flask(__name__)
app.secret_key = "change_this_secret_key"  # change for production

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

transactions_col = db.transactions
categories_col = db.categories

# -------------------------
# Utilities
# -------------------------
def parse_date(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except:
        return datetime.utcnow()

def format_date(dt):
    return dt.strftime("%Y-%m-%d")

def get_or_create_category(name):
    name = name.strip()
    if not name:
        return None
    cat = categories_col.find_one({"name": name})
    if cat:
        return cat["_id"]
    res = categories_col.insert_one({"name": name})
    return res.inserted_id

# -------------------------
# Ensure default categories at startup
# -------------------------
def ensure_defaults():
    default_cats = ["Food", "Transport", "Salary", "Rent", "Utilities", "Entertainment", "Misc"]
    for name in default_cats:
        if not categories_col.find_one({"name": name}):
            categories_col.insert_one({"name": name})

# Call once at startup
ensure_defaults()

# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    # optional filters: month (YYYY-MM) and category
    month = request.args.get("month")  # format "YYYY-MM"
    category_id = request.args.get("category")
    q = {}
    if month:
        try:
            start = datetime.strptime(month + "-01", "%Y-%m-%d")
            end = start + relativedelta(months=1)
            q["date"] = {"$gte": start, "$lt": end}
        except:
            pass
    if category_id:
        try:
            q["category_id"] = ObjectId(category_id)
        except:
            pass

    cursor = transactions_col.find(q).sort("date", -1)
    transactions = []
    for t in cursor:
        cat = categories_col.find_one({"_id": t.get("category_id")})
        transactions.append({
            "id": str(t["_id"]),
            "type": t.get("type"),
            "amount": t.get("amount"),
            "category": cat["name"] if cat else "Uncategorized",
            "date": format_date(t.get("date")),
            "description": t.get("description", "")
        })

    categories = list(categories_col.find().sort("name", 1))
    return render_template("index.html", transactions=transactions, categories=categories, selected_month=month, selected_category=category_id)

@app.route("/add", methods=["GET", "POST"])
def add_transaction():
    if request.method == "POST":
        ttype = request.form.get("type")
        amount = float(request.form.get("amount", "0") or 0)
        category_name = request.form.get("category") or request.form.get("new_category")
        description = request.form.get("description", "")
        date_str = request.form.get("date") or format_date(datetime.utcnow())
        date = parse_date(date_str)

        category_id = get_or_create_category(category_name) if category_name else None

        doc = {
            "type": ttype,
            "amount": amount,
            "category_id": category_id,
            "description": description,
            "date": date
        }
        transactions_col.insert_one(doc)
        flash("Transaction added.", "success")
        return redirect(url_for("index"))

    categories = list(categories_col.find().sort("name", 1))
    today = format_date(datetime.utcnow())
    return render_template("add_edit.html", action="Add", categories=categories, today=today)

@app.route("/edit/<id>", methods=["GET", "POST"])
def edit_transaction(id):
    try:
        obj_id = ObjectId(id)
    except:
        flash("Invalid ID", "danger")
        return redirect(url_for("index"))

    t = transactions_col.find_one({"_id": obj_id})
    if not t:
        flash("Transaction not found", "danger")
        return redirect(url_for("index"))

    if request.method == "POST":
        ttype = request.form.get("type")
        amount = float(request.form.get("amount", "0") or 0)
        category_name = request.form.get("category") or request.form.get("new_category")
        description = request.form.get("description", "")
        date_str = request.form.get("date") or format_date(datetime.utcnow())
        date = parse_date(date_str)

        category_id = get_or_create_category(category_name) if category_name else None

        transactions_col.update_one({"_id": obj_id}, {"$set": {
            "type": ttype,
            "amount": amount,
            "category_id": category_id,
            "description": description,
            "date": date
        }})
        flash("Transaction updated.", "success")
        return redirect(url_for("index"))

    categories = list(categories_col.find().sort("name", 1))
    data = {
        "id": id,
        "type": t.get("type"),
        "amount": t.get("amount"),
        "category_id": str(t.get("category_id")) if t.get("category_id") else None,
        "date": format_date(t.get("date")),
        "description": t.get("description", "")
    }
    return render_template("add_edit.html", action="Edit", categories=categories, data=data)

@app.route("/delete/<id>", methods=["POST"])
def delete_transaction(id):
    try:
        transactions_col.delete_one({"_id": ObjectId(id)})
        flash("Transaction deleted.", "success")
    except:
        flash("Could not delete transaction.", "danger")
    return redirect(url_for("index"))

@app.route("/reports", methods=["GET"])
def reports():
    # By default show current month
    month = request.args.get("month")
    if not month:
        month = datetime.utcnow().strftime("%Y-%m")
    try:
        start = datetime.strptime(month + "-01", "%Y-%m-%d")
    except:
        start = datetime.utcnow().replace(day=1)
    end = start + relativedelta(months=1)

    pipeline = [
        {"$match": {"date": {"$gte": start, "$lt": end}}},
        {"$lookup": {
            "from": "categories",
            "localField": "category_id",
            "foreignField": "_id",
            "as": "category"
        }},
        {"$unwind": {"path": "$category", "preserveNullAndEmptyArrays": True}},
        {"$project": {
            "type": 1,
            "amount": 1,
            "category": {"$ifNull": ["$category.name", "Uncategorized"]},
            "date": 1
        }},
        {"$group": {
            "_id": "$type",
            "total": {"$sum": "$amount"}
        }}
    ]
    summary_raw = list(transactions_col.aggregate(pipeline))
    totals = {"income": 0.0, "expense": 0.0}
    for r in summary_raw:
        totals[r["_id"]] = r["total"]

    # category-wise breakdown
    cat_pipeline = [
        {"$match": {"date": {"$gte": start, "$lt": end}}},
        {"$lookup": {
            "from": "categories",
            "localField": "category_id",
            "foreignField": "_id",
            "as": "category"
        }},
        {"$unwind": {"path": "$category", "preserveNullAndEmptyArrays": True}},
        {"$group": {
            "_id": {"category": {"$ifNull": ["$category.name", "Uncategorized"]}, "type": "$type"},
            "total": {"$sum": "$amount"}
        }},
        {"$project": {"category": "$_id.category", "type": "$_id.type", "total": 1, "_id": 0}},
        {"$sort": {"category": 1}}
    ]
    cat_breakdown = list(transactions_col.aggregate(cat_pipeline))

    # list transactions for the month
    tx_cursor = transactions_col.find({"date": {"$gte": start, "$lt": end}}).sort("date", -1)
    txs = []
    for t in tx_cursor:
        cat = categories_col.find_one({"_id": t.get("category_id")})
        txs.append({
            "id": str(t["_id"]),
            "type": t.get("type"),
            "amount": t.get("amount"),
            "category": cat["name"] if cat else "Uncategorized",
            "date": format_date(t.get("date")),
            "description": t.get("description", "")
        })

    return render_template("reports.html", month=month, totals=totals, cat_breakdown=cat_breakdown, transactions=txs)

@app.route("/export", methods=["GET"])
def export_csv():
    # export filtered by optional month
    month = request.args.get("month")
    q = {}
    if month:
        try:
            start = datetime.strptime(month + "-01", "%Y-%m-%d")
            end = start + relativedelta(months=1)
            q["date"] = {"$gte": start, "$lt": end}
        except:
            pass

    cursor = transactions_col.find(q).sort("date", -1)
    rows = []
    for t in cursor:
        cat = categories_col.find_one({"_id": t.get("category_id")})
        rows.append({
            "date": format_date(t.get("date")),
            "type": t.get("type"),
            "amount": t.get("amount"),
            "category": cat["name"] if cat else "Uncategorized",
            "description": t.get("description", "")
        })

    # create CSV in-memory
    df = pd.DataFrame(rows)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    return send_file(io.BytesIO(csv_buffer.getvalue().encode('utf-8')),
                     mimetype='text/csv',
                     as_attachment=True,
                     download_name=f"transactions_{month or 'all'}.csv")

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
