from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Employee(BaseModel):
    name:str
    age:int
    company:str = 'PwC'   # assigning default value
    pluxee_card: Optional[int] = None
    email: EmailStr
    yoe:int=Field(gt=0,lt=20, description='Overall years of experience') # built-in validation

# pydantic performs implicit coercion, for example it tries to typecast boolean or string to number for age field below
new_emp_details = {"name":"Prachi", "age":26,"email":"prachi@gmail.com","yoe":4}

emp1 = Employee(**new_emp_details)
#emp1 object is of type Pydantic object

print(emp1)

print(type(emp1.age))

emp1_dict = emp1.model_dump()

print(type(emp1),type(emp1_dict))

emp1_json = emp1.model_dump_json()

print(type(emp1_json))