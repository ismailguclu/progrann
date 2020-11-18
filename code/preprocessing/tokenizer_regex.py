import re

# FILE 124-03.xml is funny, because the doctor's name is ismail

REGEX_LIST_PREFIX = [
    r"(?:-)(?=[0-9])",                          # splits on hyphen at the start
    r"(?:--)(?=[0-9])",                         # splits on double hyphen at the start
    r"(?:----)(?=[0-9])",                       # splits on 4 hyphens at the start
    r"(?:~)(?=[0-9])",                          # splits on tilde at the start

    r"(?:-)(?=[A-z])",                          # splits on hyphen at the start
    r"(?:/)(?=[A-z])",                          # splits on forward slash at the start
    r"(?:\\)(?=[A-z])",                         # splits on backward slash at the start
]

REGEX_LIST_INFIX = [
    r"(?<=[0-9])(?:-)(?=[A-z])",                # splits on hyphen between number and letter
    r"(?<=[0-9])(?:.)(?=[A-z])",                # splits on dot between number and letter
    r"(?<=[0-9])(?:\^)(?=[A-z])",               # splits on caret between number and letter
    r"(?<=[0-9])(?:=&)(?=[A-z])",               # splits on =/ampersand between number and letter
    r"(?<=[0-9])(?:=)(?=[0-9])",                # splits on = between number and number
    r"(?<=[0-9])(?:;)(?=[A-z])",                # splits on semi-colon between number and letter
    r"(?<=[0-9])(?:&#)(?=[0-9])",               # splits on ampersand/hashtag number and number
    r"(?<=[0-9])(?:.<BR>--)(?=[A-z])",          # splits on special structure between number and letter
    r"(?<=[0-9])(?:\)--)(?=[A-z])",             # splits on round bracket/double hyphen between number and letter
    r"(?<=[0-9])(?:\))(?=[0-9])",               # splits on round bracket between number and number
    r"([0-9][0-9]\/[0-9]+\/[0-9]+)",            # splits on date (XX/XX/XXXX) structure
    r"(?<=[0-9])(?:,)(?=[0-9])",                # splits on comma between number and number
    r"(?<=[0-9])(?:.)(?=[0-9])",                # splits on dot between number and number
    r"(?<=[0-9])(?:->)(?=[0-9])",               # splits on -> between number and number

    r"(?<=[A-z])(?:-)(?=[A-z])",                # splits on hyphen between letter and letter
    r"(?<=[A-z])(?::)(?=[0-9])",                # splits on colon between letter and number
    r"(?<=[A-z])(?:;)(?=[0-9])",                # splits on semi-colon between letter and number
    r"(?<=[A-z])(?:\^)(?=[0-9])",               # splits on caret between letter and number
    r"(?<=[A-z])(?:.)(?=[0-9])",                # splits on dot between letter and number
    r"(?<=[A-z])(?:#)(?=[0-9])",                # splits on hashtag between letter and number
    r"(?<=[A-z])(?:----)(?=[0-9])",             # splits on four hyphens between letter and number
    r"(?<=[A-z])(?:----)(?=[A-z])",             # splits on four hyphens between letter and letter
    r"(?<=[A-z])(?:#)(?=[A-z])",                # splits on hashtag between letter and letter
    r"(?<=[A-z])(?:\()(?=[A-z])",               # splits on round bracket between letters
    r"(?<=[A-z])(?:\.\-)(?=[A-z])",             # splits on dot/hyphen between letters
    r"(?<=[A-z])(?:_\()(?=[A-z])",              # splits on underscore/round bracket between letters
    r"(?<=[A-z])(?:#:)(?=[0-9])",               # splits on hashtag/colon between letter and number
    r"(?<=[A-z])(?:&#)(?=[0-9])",               # splits on ampersand/hashtag letter and number
    r"(?<=[A-z])(?:<)(?=[0-9])",                # splits on smaller than between letter and number   
    r"(?<=[A-z])(?:.)((?:_)+)(?=[A-z])",        # splits on dot/multiple underscore between letter and letter
    r"(?<=[A-z])(?:\))(?=[A-z])",               # splits on round bracket between letter and letter
    r"(?<=[0-9])(?:\),)(?=[A-z])",              # splits on round bracket/comma between number and letter
]

REGEX_LIST_SUFFIX = [
    r"(?<=[0-9])(?:.doc)",                      # splits on .doc at the end
    r"(?<=[0-9])(?:-)",                         # splits on hyphen at the end
    r"(?<=[0-9])(?:yo)",                        # splits on yo at the end
    r"(?<=[0-9])(?:=)",                         # splits on = at the end
    r"(?<=[0-9])(?::-)",                        # splits on colon/hyphen at the end
    r"(?<=[0-9])(?:\)\+)",                      # splits on round bracket/plus
    r"(?<=[0-9])(?:\)-)",                       # splits on round bracket/dash at the end

    r"(?<=[A-z])(?:--)",                        # splits on double hyphen at the end
    r"(?<=[A-z])(?:---)",                       # splits on triple hyphen at the end
    r"(?<=[A-z])(?:/)",                         # splits on forward slash at the end
    r"(?<=[A-z])(?:-)",                         # splits on dash at the end
    r"(?<=[A-z])(?:=)",                         # splits on = at the end
]