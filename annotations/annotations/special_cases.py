from spacy.symbols import ORTH

SPECIAL_CASES_LIST = [
    ("49yoRHM", [{ORTH: "49"}, {ORTH: "yo"}, {ORTH: "RHM"}]),
    ("36052413AGH", [{ORTH: "36052413"}, {ORTH: "AGH"}]), 
    ("9654616MC", [{ORTH: "9654616"}, {ORTH: "MC"}]),
    ("855930.do", [{ORTH: "855930"}, {ORTH: ".do"}]),     
    ("502491.do", [{ORTH: "502491"}, {ORTH: ".do"}]),
    ("May.", [{ORTH: "May"}, {ORTH: "."}]),
    ("CABG6/95", [{ORTH: "CABG"}, {ORTH: "6/95"}]),
    ("WKJ/3454", [{ORTH: "WKJ"}, {ORTH: "/"}, {ORTH: "3454"}]),
    ("FAT", [{ORTH: "F"}, {ORTH: "A"}, {ORTH: "T"}]),
    ("STTh", [{ORTH: "S"}, {ORTH: "T"}, {ORTH: "Th"}]),
    ("x13650", [{ORTH: "x"}, {ORTH: "13650"}]),
    ("ThisRoberta", [{ORTH: "This"}, {ORTH: "Roberta"}]),
    ("GambiaHome", [{ORTH: "Gambia"}, {ORTH: "Home"}]),
    ("ALMarital", [{ORTH: "AL"}, {ORTH: "Marital"}]),
    ("SupervisorSupport", [{ORTH: "Supervisor"}, {ORTH: "Support"}]),
    ("JaffreyMarital", [{ORTH: "Jaffrey"}, {ORTH: "Marital"}]),
    ("s.", [{ORTH: "s"}, {ORTH: "."}]),
    ("Mt.", [{ORTH: "Mt"}, {ORTH: "."}]),
    ("inhartsville", [{ORTH: "in"}, {ORTH: "hartsville"}]),
    ("ELLENMRN", [{ORTH: "ELLEN"}, {ORTH: "MRN"}]),
    ("X1-1335)", [{ORTH: "X"}, {ORTH: "1-1335"}, {ORTH: ")"}]),
    ("HomeEmergency", [{ORTH: "Home"}, {ORTH: "Emergency"}]),
    ("seeB.", [{ORTH: "see"}, {ORTH: "B."}]),
    ("Jamiesonfor", [{ORTH: "Jamieson"}, {ORTH: "for"}]),
    ("TMSIII", [{ORTH: "TMS"}, {ORTH: "III"}]),
    ("NYSH-Ed", [{ORTH: "NYSH"}, {ORTH: "-"}, {ORTH: "Ed"}]),
    ("PIMA", [{ORTH: "P"}, {ORTH: "I"}, {ORTH: "M"}, {ORTH: "A"}]),
    ("Ayala-Skiff", [{ORTH: "Ayala"}, {ORTH: "-"}, {ORTH: "Skiff"}]),
    ("abiological", [{ORTH: "a"}, {ORTH: "biological"}]),
    ("Hospital0021", [{ORTH: "Hospital"}, {ORTH: "0021"}]),
    ("AvenueKigali,", [{ORTH: "Avenue"}, {ORTH: "Kigali"}, {ORTH: ","}]),
    ("47798497-045-1949", [{ORTH: "47798"}, {ORTH: "497-045-1949"}]),
    ("Sept.", [{ORTH: "Sept"}, {ORTH:"."}]),
    ("Nov.", [{ORTH: "Nov"}, {ORTH: "."}]),
    ("Mississippi-did", [{ORTH: "Mississippi"}, {ORTH: "-"}, {ORTH: "did"}]),
    ("R.", [{ORTH: "R"}, {ORTH: "."}]),
    ("W.", [{ORTH: "W"}, {ORTH: "."}]),
    ("M.", [{ORTH: "M"}, {ORTH: "."}]),
    ("2060'S.", [{ORTH: "2060'S"}, {ORTH: "."}]),
    ("M/W/F.", [{ORTH: "M"}, {ORTH: "/"}, {ORTH: "W"}, {ORTH: "/"}, {ORTH: "F"}, {ORTH: "."}]),
    ("Inc.", [{ORTH: "Inc"}, {ORTH: "."}]),
    ("TThSa", [{ORTH: "T"}, {ORTH: "Th"}, {ORTH: "Sa"}]),
    ("Indonesian-speaking", [{ORTH: "Indonesian"}, {ORTH: "-"}, {ORTH: "speaking"}]),
    ("EFSIII", [{ORTH: "EFS"}, {ORTH: "III"}]),
    ("HospitalEmergency", [{ORTH: "Hospital"}, {ORTH: "Emergency"}]),
    ("aveterinary", [{ORTH: "a"}, {ORTH: "veterinary"}]),
    ("feb.<BR>c/o", [{ORTH: "feb"}, {ORTH: "."}, {ORTH: "<BR>"}, {ORTH: "c/o"}]),
    ("2095<BR>--bp", [{ORTH: "2095"}, {ORTH: "<BR>"}, {ORTH: "--"}, {ORTH: "bp"}]),
    ("CenterEmergency", [{ORTH: "Center"}, {ORTH: "Emergency"}]),
    ("Ishidas", [{ORTH: "Ishida"}, {ORTH: "s"}]),
    ("TTS", [{ORTH: "T"}, {ORTH: "T"}, {ORTH: "S"}]),
    ("HospitalStaff", [{ORTH: "Hospital"}, {ORTH: "Staff"}]),
    ("MWFS", [{ORTH: "M"}, {ORTH: "W"}, {ORTH: "F"}, {ORTH: "S"}]),
    ("MWF", [{ORTH: "M"}, {ORTH: "W"}, {ORTH: "F"}]),
    ("CMHuck", [{ORTH: "CMH"}, {ORTH: "uck"}]),
    ("OldhamChief", [{ORTH: "Oldham"}, {ORTH: "Chief"}]),
    ("Filenesas", [{ORTH: "Filenes"}, {ORTH: "as"}]),
    ("NVHBLOOD", [{ORTH: "NVH"}, {ORTH: "BLOOD"}]),
    ("Appraiser-currently", [{ORTH: "Appraiser"}, {ORTH: "-"}, {ORTH: "currently"}]),
    ("HospitalAdmission", [{ORTH: "Hospital"}, {ORTH: "Admission"}]),
    ("HospitalIntern", [{ORTH: "Hospital"}, {ORTH: "Intern"}]),
    ("QSun,Mon,Wed,Fri", [{ORTH: "Q"}, {ORTH: "Sun"}, {ORTH: ","}, {ORTH: "Mon"}, {ORTH: ","}, {ORTH: "Wed"}, {ORTH: ","}, {ORTH: "Fri"}]),
    ("atNorth", [{ORTH: "at"}, {ORTH: "North"}]),
    ("GarzaRHN", [{ORTH: "Garza"}, {ORTH: "RHN"}]),
    ("RHNThe", [{ORTH: "RHN"}, {ORTH: "The"}]),
    ("appt-JHThe", [{ORTH: "appt"}, {ORTH: "-"}, {ORTH: "JH"}, {ORTH: "The"}]),
    ("LittletonColonoscopy", [{ORTH: "Littleton"}, {ORTH: "Colonoscopy"}]),
    ("NorthProblems", [{ORTH: "North"}, {ORTH: "Problems"}]),
    ("RNHHe", [{ORTH: "RNH"}, {ORTH: "He"}]),
    ("FredMR", [{ORTH: "Fred"}, {ORTH: "MR"}]),
    ("Amer..", [{ORTH: "Amer."}, {ORTH: "."}]),
    ("V.,M.D.,PH.D.", [{ORTH: "V."}, {ORTH: ","}, {ORTH: "M.D."}, {ORTH: ","}, {ORTH: "PH.D."}]),
    ("Valdivia.he", [{ORTH: "Valdivia"}, {ORTH: "."}, {ORTH: "he"}]),
    ("qWednesday", [{ORTH: "q"}, {ORTH: "Wednesday"}]),
    ("OgradyIMC", [{ORTH: "Ogrady"}, {ORTH: "IMC"}]),
    ("Xavier;P::", [{ORTH: "Xavier"}, {ORTH: ";"}, {ORTH: "P::"}]),
    ("qthursday", [{ORTH: "q"}, {ORTH: "thursday"}]),
    ("qwednesday", [{ORTH: "q"}, {ORTH: "wednesday"}]),
    ("PROMPTCAREIMA", [{ORTH: "PROMPTCARE"}, {ORTH: "IMA"}]),
    ("Mayof", [{ORTH: "May"}, {ORTH: "of"}]),
    ("MIMA", [{ORTH: "M"}, {ORTH: "IMA"}]),
    ("MIMAIMA", [{ORTH: "MIMA"}, {ORTH: "IMA"}]),
    ("QSunday", [{ORTH: "Q"}, {ORTH: "Sunday"}]),
    ("QSundayAM", [{ORTH: "Q"}, {ORTH: "Sunday"}, {ORTH: "AM"}]),
    ("U.,RNC", [{ORTH: "U"}, {ORTH: "."}, {ORTH: ","}, {ORTH: "RNC"}]),
    ("NOTE.GCC", [{ORTH: "NOTE"}, {ORTH: "."}, {ORTH: "GCC"}]),
    ("GCCfor", [{ORTH: "GCC"}, {ORTH: "for"}]),
    ("HCC3", [{ORTH: "HCC"}, {ORTH: "3"}]),
    ("PLAN88F", [{ORTH: "PLAN"}, {ORTH: "88"}, {ORTH: "F"}]),
    ("20626842267", [{ORTH: "2062"}, {ORTH: "6842267"}]),
]