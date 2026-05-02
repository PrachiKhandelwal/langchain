from typing import TypedDict

class Employee(TypedDict):
    name:str
    age:int
    
new_emp: Employee = {
    'name':'pra',
    'age':24
}

print(new_emp)