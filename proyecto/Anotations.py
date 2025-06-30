import pandas as pd
import os 
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
        # Obtener el path absoluto a la carpeta 'archivos_generados'
        base_dir = os.path.dirname(os.path.dirname(__file__))  # Subir un nivel desde 'proyecto'
        target_dir = os.path.join(base_dir, "archivos_generados")
        
        # Crear la carpeta si no existe
        os.makedirs(target_dir, exist_ok=True)

        # Construir la ruta completa al archivo
        full_path = os.path.join(target_dir, filename)

        # Guardar el archivo
        self.anotations.to_csv(full_path, index=False)
    
    def load(self, filename):
        import os
        import pandas as pd

        # Si filename ya es una ruta absoluta, usala tal cual.
        if not os.path.isabs(filename):
            # Si es relativa, convertila en ruta absoluta desde este archivo.
            base_dir = os.path.dirname(__file__)  # carpeta actual donde está la clase
            full_path = os.path.abspath(os.path.join(base_dir, filename))
        else:
            full_path = filename

        if not os.path.exists(full_path):
            print(f"Archivo no encontrado en: {full_path}")
            return

        self.anotations = pd.read_csv(full_path)

    def __str__(self):
        return str(self.anotations)