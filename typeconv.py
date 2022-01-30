def typeconvo(newdf):
    newdf.Sex = newdf.Sex.apply(Sex)
    newdf.ChestPainType = newdf.ChestPainType.apply(ChestPainType)
    newdf.RestingECG = newdf.RestingECG.apply(RestingECG)
    newdf.ExerciseAngina = newdf.ExerciseAngina.apply(ExerciseAngina)
    newdf.ST_Slope = newdf.ST_Slope.apply(ST_Slope)
    return newdf

def Sex(value):
    if value == 'M' or value == "Male":
        return 1
    elif value == 'F' or value =="Female":
        return 2
def ChestPainType(value):
    if value == 'Asymptomatic' or value == 'ASY':
        return 1
    elif value == 'Non-Anginal Pain' or value == 'NAP':
        return 2
    elif value == 'Atypical Angina' or value == 'ATA':
        return 3
    elif value == 'Typical Angina' or value == 'TA':
        return 4
def RestingECG(value):
    if value == 'LVH' or value == "Showing probable or definite left ventricular hypertrophy by Estes' criteria":
        return 1
    elif value == 'ST' or value == "Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)":
        return 2
    else:
        return 3
def ExerciseAngina(value):
    if value == 'Y' or value == "Yes":
        return 1
    elif value == 'N' or value == "No":
        return 2
def ST_Slope(value):
    if value == 'Up' or value == 'Upsloping':
        return 1
    elif value =='Downsloping' or value == 'Down':
        return 2
    else:
        return 3
