from setuptools import find_packages,setup
from typing import List


def getrequirements(file_path:str)->List[str]:
    '''
    This function will return list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements= file_obj.readlines()
        requirements= [req.replace("\n","") for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements        

setup(
    name='healthplusclinic',
    version ='1.0',
    author='subash',
    author_email='ssunuwar33@gmail.com',
    packages=find_packages(),
    install_requires=getrequirements('requirements.txt')

    
)