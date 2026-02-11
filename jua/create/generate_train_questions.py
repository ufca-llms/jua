from beir.datasets.data_loader import GenericDataLoader
import dotenv
from pydantic import BaseModel, Field
from openai import OpenAI
import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from tqdm import tqdm

dotenv.load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

class Question(BaseModel):
    question: str = Field(description="A query gerada com o padrão \"keyword1 keyword2 keyword3\"")

prompter = llm.with_structured_output(Question)

def generate_question(ementa: str) -> Question:
    return prompter.invoke([
        SystemMessage(content="Elabore uma query de busca curta baseada em keywords que possa ser usada para encontrar a seguinte ementa judicial:"),
        HumanMessage(content=ementa)
    ])

if __name__ == "__main__":
    dataset_path = "./jua-dataset"
    corpus, queries, qrels = GenericDataLoader(
        corpus_file=f"{dataset_path}/corpus.jsonl", 
        query_file=f"{dataset_path}/queries.jsonl", 
        qrels_file=f"{dataset_path}/qrels/train.tsv").load_custom()

    for qid, query in tqdm(queries.items()):
        question = generate_question(query).question
        qry_dict = {"_id": qid, "text": question}
        # ensure ascii encoding and ignore errors        question_ascii = question.encode("ascii", errors="ignore").decode("ascii")
        with open(f"{dataset_path}/queries_with_questions_new.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(qry_dict, ensure_ascii=False) + "\n")        
   

# python3 -m jua.create.generate_train_questions