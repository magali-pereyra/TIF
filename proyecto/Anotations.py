import pandas as pd
class Anotaciones():
    def __init__(self, onset=None, duration=None, description=None):
        if onset and duration and description:
            self.anotations = pd.DataFrame({
                "onset": onset,
                "duration": duration,
                "description": description
            })
        else:
            self.anotations = pd.DataFrame(columns=["onset", "duration", "description"])

    def add(self, onset, duration, description):
        if description in self.anotations["description"].values:
            print(f"Error! Ya existe una anotación con la descripción: {description}")
            return  # No agrega la nueva fila

        nueva_anotacion = pd.DataFrame([{
            "onset": onset,
            "duration": duration,
            "description": description
        }])
        self.anotations = pd.concat([self.anotations, nueva_anotacion], ignore_index=True)
    
    def remove(self, description):
            # Elimina todas las filas cuyo valor en "description" coincida exactamente
            self.anotations = self.anotations[self.anotations["description"] != description].reset_index(drop=True)

    def get_anotations(self, description=None):
        if description is None:
            return self.anotations
        else:
            return self.anotations[self.anotations["description"] == description].reset_index(drop=True)
    
    def find(self, description):
        return self.anotations[self.anotations["description"]==description]
    
    def save(self,filename):
        self.anotations.to_csv(filename,index=False)
    
    def load(self,filename):
        self.anotations=pd.read_csv(filename)
    
    def __str__(self):
        return str(self.anotations)