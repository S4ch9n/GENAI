from pydantic import BaseModel ,Field
from typing import Optional

class Student(BaseModel):  # define schema

    name: str = 'nitish'
    age : Optional[int] = None  # name must be string
    cgpa : float = Field(gt = 0 , lt = 10 , default = 10 , description='A decimal representing the cgpa of student')

new_student = {'age': 32 , 'cgpa' : 7.4}  # raw dictionary

student = Student(**new_student)  # validate and create object
student_dict = dict(student)

print(student_dict['age'])


student__json = student.model_dump_json()

