import pandas as pd
import json
import os
from datetime import datetime
import io
import re

from sqlalchemy import create_engine, inspect
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

import plotly.express as px

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# =========================
# Visualization Detection
# =========================
def should_visualize(question):
    keywords = [
        "chart", "graph", "plot", "visualize", "trend", "compare",
        "distribution", "top", "bottom", "highest", "lowest",
        "month", "year", "sales", "revenue", "customer"
    ]
    return any(word in question.lower() for word in keywords)


# =========================
# SQL Result Parsing
# =========================
def sql_result_to_dataframe(results):
    """
    Converts SQLAlchemy results into a dataframe with two columns for plotting.
    Tries to keep the first column as x/label and second column as y/value.
    """
    if not results:
        return None

    # If results are list of tuples
    if isinstance(results, list) and all(isinstance(r, (list, tuple)) for r in results):
        try:
            df = pd.DataFrame(results)
            if df.shape[1] < 2:
                return None
            # Rename columns to standard names for chart function
            df.columns = ["label", "value"]
            return df
        except:
            return None

    # If results are just one tuple
    if isinstance(results, tuple) and len(results) == 2:
        return pd.DataFrame([results], columns=["label", "value"])

    # Fallback: try string parsing (legacy)
    labels, values = [], []
    for row in str(results).split("\n"):
        match = re.search(r"\('([^']+)',\s*([\d\.]+)\)", row)
        if match:
            labels.append(match.group(1))
            values.append(float(match.group(2)))

    if labels and values:
        return pd.DataFrame({"label": labels, "value": values})

    return None



# =========================
# Chart Builder (Second code version)
# =========================
def create_chart(df, question):
    if df is None or df.shape[1] < 2:
        return None

    x = df.columns[0]
    y = df.columns[1]

    q = question.lower()

    # Trend chart
    if any(word in q for word in ["trend", "month", "year"]):
        return px.line(df, x=x, y=y, title="Trend Analysis")

    # Distribution chart
    if any(word in q for word in ["distribution", "share"]):
        return px.pie(df, names=x, values=y, title="Distribution")

    # Ranking / Top / Bottom chart
    if any(word in q for word in ["top", "highest", "lowest", "rank"]):
        return px.bar(df, x=y, y=x, orientation="h", title="Ranking")

    # Default bar chart
    return px.bar(df, x=x, y=y, title="Analysis")


# =========================
# File Handling
# =========================
def read_file(uploaded_file):
    try:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file), None
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file), None
        return None, "Unsupported file type"
    except Exception as e:
        return None, str(e)


def create_sql_database(data, table_name="business_data"):
    try:
        engine = create_engine("sqlite:///temp_business_data.db")
        data.to_sql(table_name, engine, if_exists="replace", index=False)

        inspector = inspect(engine)
        columns = inspector.get_columns(table_name)
        schema = [(col["name"], str(col["type"])) for col in columns]

        db = SQLDatabase(engine)
        return db, schema, None

    except Exception as e:
        return None, None, str(e)


# =========================
# Chat Persistence
# =========================
def save_chat_to_file(chat_data, filename):
    os.makedirs("saved_chats", exist_ok=True)

    clean_messages = []
    for msg in chat_data["messages"]:
        m = msg.copy()
        m.pop("chart", None)
        clean_messages.append(m)

    chat_data["messages"] = clean_messages

    with open(os.path.join("saved_chats", filename), "w") as f:
        json.dump(chat_data, f, indent=2)

    return True


def load_chat_from_file(filename):
    path = os.path.join("saved_chats", filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def get_saved_chats():
    if not os.path.exists("saved_chats"):
        return []
    return sorted(
        [f for f in os.listdir("saved_chats") if f.endswith(".json")],
        reverse=True
    )


def delete_saved_chat(filename):
    path = os.path.join("saved_chats", filename)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


# =========================
# AI ANALYSIS AGENT
# =========================
class AnalysisAgent:
    def __init__(self, database, api_key):
        self.database = database
        self.chat_history = []

        # setup the AI model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0
        )

        # SQL generation prompt (strict, system prompt restored)
        sql_template = """You are an AI Data Analyst connected to a SQL database.

STRICT RULES:
1. You MUST always check the database schema provided below.
2. You MUST identify table name(s) and column names before generating SQL.
3. You MUST infer column names by meaning, not exact wording.
4. You MUST generate ONE valid SELECT SQL query to answer the question.
5. You MUST NOT use general knowledge or assumptions.

IMPORTANT COLUMN MATCHING RULE:
- If a column name is not an exact match to the user's wording, you MUST infer the correct column name by meaning.
- Example: "customer", "buyer", or "client" may refer to columns like customername, client_name, or buyer_name.
- Always choose the closest matching column based on semantic meaning.
- Never fail just because the column name is not an exact text match.

BEHAVIOR:
- Before generating SQL, identify which table and which columns are relevant.
- If the question asks for numbers (totals, averages, trends), use aggregate functions (SUM, AVG, COUNT).
- If the question asks "why" something happened, query historical data, trends, and comparisons.
- If the question asks for suggestions or growth strategies, query relevant performance metrics.
- For vague questions like "which product to focus on", query top performers by revenue/sales.

CHART DATA REQUIREMENT:
- If the question involves comparisons, rankings, trends, distributions, or top/bottom items, return results suitable for visualization.
- Use column aliases that are clear: SELECT category as label, SUM(amount) as value
- Order results logically (DESC for rankings, chronologically for trends)
- Limit to reasonable numbers for charts (top 10, top 5, etc.)

OUTPUT RULES:
- Return ONLY a valid SQL SELECT query
- No explanations, no markdown, no backticks
- Just the SQL query that can be executed directly
- Use proper SQL syntax for SQLite

Never hallucinate column names. Never answer without checking the schema first.

Database schema:
{schema}

Question: {question}

SQL Query:"""

        self.sql_prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template=sql_template
        )
        self.sql_chain = LLMChain(llm=self.llm, prompt=self.sql_prompt)

        # Analysis prompt
        analysis_template = """You are a business analyst. Use this data to answer the question.

Data:
{results}

SQL:
{sql}

Question: {question}

Provide your answer in 3 sections:
1. What I Found
2. Why It Matters
3. Recommendations"""

        self.analysis_prompt = PromptTemplate(
            input_variables=["question", "sql", "results"],
            template=analysis_template
        )
        self.analysis_chain = LLMChain(llm=self.llm, prompt=self.analysis_prompt)

    def analyze(self, question):
        # get database schema
        schema_text = ""
        tables = self.database.get_usable_table_names()
        for table in tables:
            schema_text += self.database.get_table_info([table])

        # generate SQL query
        sql_response = self.sql_chain.invoke({
            "schema": schema_text,
            "question": question
        })
        raw_sql = sql_response.get("text", "").strip()
        sql_query = raw_sql.replace("sql", "").replace("", "").strip()
        if sql_query.startswith("SQL:"):
            sql_query = sql_query[4:].strip()
        if not sql_query.endswith(";"):
            sql_query += ";"

        # run the query
        try:
            results = self.database.run(sql_query)
        except Exception as e:
            return {
                "status": "error",
                "analysis": f"Error running query: {e}",
                "chart": None,
                "sql_results": None
            }

        # create chart if needed
        chart = None
        df = sql_result_to_dataframe(results)
        if df is not None and should_visualize(question):
            chart = create_chart(df, question)

        # generate analysis
        analysis_response = self.analysis_chain.invoke({
            "question": question,
            "sql": sql_query,
            "results": results
        })
        analysis = analysis_response.get("text", "")

        # save to history
        self.chat_history.append({
            "question": question,
            "result": results,
            "analysis": analysis,
            "timestamp": datetime.now().strftime("%I:%M %p")
        })

        return {
            "sql_results": results,
            "analysis": analysis,
            "chart": chart,
            "status": "success"
        }

    def get_chat_history(self):
        return self.chat_history

    def clear_history(self):
        self.chat_history = []


# =========================
# Generate PDF report
# =========================
def generate_pdf_report(question, sql_results, analysis):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    story = []
    story.append(Paragraph("Business Data Analysis Report", styles["Heading1"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Question:</b> {question}", styles["BodyText"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Results:</b><br/>{sql_results}", styles["BodyText"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Analysis:</b><br/>{analysis}", styles["BodyText"]))

    doc.build(story)
    buffer.seek(0)
    return buffer
