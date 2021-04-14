import re

# FILE 124-03.xml is funny, because the doctor's name is ismail

REGEX_LIST_PREFIX = [
    r"(?:-)(?=[0-9])",                          
    r"(?:--)(?=[0-9])",                         
    r"(?:----)(?=[0-9])",                       
    r"(?:~)(?=[0-9])",                          

    r"(?:-)(?=[A-z])",                          
    r"(?:/)(?=[A-z])",                          
    r"(?:\\)(?=[A-z])",                         
]

REGEX_LIST_INFIX = [
    r"(?<=[0-9])(?:-)(?=[A-z])",                
    r"(?<=[0-9])(?:.)(?=[A-z])",                
    r"(?<=[0-9])(?:\^)(?=[A-z])",               
    r"(?<=[0-9])(?:=&)(?=[A-z])",               
    r"(?<=[0-9])(?:=)(?=[0-9])",                
    r"(?<=[0-9])(?:;)(?=[A-z])",                
    r"(?<=[0-9])(?:&#)(?=[0-9])",               
    r"(?<=[0-9])(?:.<BR>--)(?=[A-z])",          
    r"(?<=[0-9])(?:\)--)(?=[A-z])",             
    r"(?<=[0-9])(?:\))(?=[0-9])",               
    r"([0-9][0-9]\/[0-9]+\/[0-9]+)",            
    r"(?<=[0-9])(?:,)(?=[0-9])",      
    r"(?<=[0-9])(?:;)(?=[0-9])",           
    r"(?<=[0-9])(?:\.)(?=[0-9])",                # splits on dot between number and number
    r"(?<=[0-9])(?:->)(?=[0-9])",       
    r"(?<=[0-9])(?:\/)(?=[0-9])",        

    r"(?<=[A-z])(?:-)(?=[A-z])",                
    r"(?<=[A-z])(?::)(?=[0-9])",                
    r"(?<=[A-z])(?:;)(?=[0-9])",                
    r"(?<=[A-z])(?:\^)(?=[0-9])",               
    r"(?<=[A-z])(?:.)(?=[0-9])",                
    r"(?<=[A-z])(?:#)(?=[0-9])",                
    r"(?<=[A-z])(?:----)(?=[0-9])",             
    r"(?<=[A-z])(?:----)(?=[A-z])",             
    r"(?<=[A-z])(?:#)(?=[A-z])",                
    r"(?<=[A-z])(?:\()(?=[A-z])",               
    r"(?<=[A-z])(?:\.\-)(?=[A-z])",             
    r"(?<=[A-z])(?:_\()(?=[A-z])",              
    r"(?<=[A-z])(?:#:)(?=[0-9])",               
    r"(?<=[A-z])(?:&#)(?=[0-9])",               
    r"(?<=[A-z])(?:<)(?=[0-9])",                   
    r"(?<=[A-z])((?:_)+)(?=[A-z])",        
    r"(?<=[A-z])(?:\))(?=[A-z])",               
    r"(?<=[0-9])(?:\),)(?=[A-z])",              
]

REGEX_LIST_SUFFIX = [
    r"(?<=[0-9])(?:.doc)",                      
    r"(?<=[0-9])(?:-)",                         
    r"(?<=[0-9])(?:yo)",   
    r"(?<=[0-9])(?:yM)", 
    r"(?<=[0-9])(?:y/o)",                      
    r"(?<=[0-9])(?:=)",                         
    r"(?<=[0-9])(?::-)",                        
    r"(?<=[0-9])(?:\)\+)",                      
    r"(?<=[0-9])(?:\)-)",                       

    r"(?<=[A-z])(?:--)",                        
    r"(?<=[A-z])(?:---)",                       
    r"(?<=[A-z])(?:/)",                         
    r"(?<=[A-z])(?:-)",                         
    r"(?<=[A-z])(?:=)",                         
]