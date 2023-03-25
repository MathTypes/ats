import os
import sys
import pandas as pd
from sqlalchemy import create_engine
import ipywidgets as widgets
from IPython.display import display
from IPython.display import update_display
from IPython.display import display_pretty


SQLALCHEMY_DATABASE_URI = os.environ.get('SQLALCHEMY_DATABASE_URI')
engine = create_engine(SQLALCHEMY_DATABASE_URI)

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from api import GPT, Example

#Construct GPT-3-instruct instance, add instruction and examples
gpt = GPT(engine="davinci-instruct-beta",
          temperature=0.3,
          max_tokens=200)
gpt.add_instruction('Given an input question, respond with syntactically correct PostgreSQL.')

gpt.add_example(Example('select columns from users table', 
                        'select id, email, dt, plan_type, created_on, updated_on from users'))
gpt.add_example(Example('select columns from the charges table', 
                        'select amount, amount_refunded, created, customer_id, status from charges'))
gpt.add_example(Example('select columns from the customers table', 
                        'select created, email, id from customers'))
output = gpt.submit_request(inp.value)
    result = output['choices'][0].text
    query = result.split('output:')[1]
    print ('\033[1mGPT-3 Response:\033[0m ' + query)                        