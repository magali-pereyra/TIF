import pandas as pd
class Info:
    """Clase para almacenar información acerca del registro de datos.
    Esta clase se comporta como un diccionario.
    """
    def __init__(self, ch_names=None, ch_types="unknown", bads=None, sfreq=512,
                description="No data", experimenter="No data", subject_info="No data"):
        """Genera un objeto Info().
            Parameters
            ----------
            ch_names : list of str, optional
                Lista con los nombres de los canales.
            ch_types : str or list of str, optional
                Tipo de cada canal (ej: 'eeg', 'ecg', etc.) o un único tipo para todos.
            bads : list of str, optional
                Lista de canales marcados como "malos".
            sfreq : float, optional
                Frecuencia de muestreo en Hz (por defecto 512).
            description : str, optional
                Descripción del registro de datos.
            experimenter : str, optional
                Nombre del experimentador.
            subject_info : dict, optional
                Información adicional del sujeto.
            Raises
            ------
            ValueError
                Si 'ch_names' y 'ch_types' no tienen la misma longitud."""
        
        self.ch_names = ch_names
        self.sfreq = sfreq
        self.description = description
        self.experimenter = experimenter
        self.subject_info = subject_info
        self.bads = bads if bads is not None else []

        # Expandir ch_types si es una cadena
        if isinstance(ch_types, str):
            if ch_names is not None:
                self.ch_types = [ch_types] * len(ch_names)
            else:
                self.ch_types = []
        else:
            self.ch_types = ch_types

        # Verificar que las longitudes coincidan
        if self.ch_names is not None and len(self.ch_names) != len(self.ch_types):
            raise ValueError("La longitud de 'ch_names' y 'ch_types' debe ser la misma.")
    
    ### CONTAINS
    def __contains__(self, key):
        """Permite usar 'in' para verificar si una clave existe."""
        return key in self.__dict__
    
    ### GET ITEM
    def __getitem__(self, key):
        return self.__dict__.get(key, f"[{key}] no encontrado")
    
    ### LEN
    def __len__(self):
        """Devuelve la cantidad de claves con valores útiles."""
        return len([v for v in self.__dict__.values()])

    ### GET
    def get(self,key):
        """Devuelve el valor asociado a 'key'"""
        return self[key]
    
    ### KEYS
    def keys(self):
        return self.__dict__.keys()
    import pandas as pd

    ### ITEMS
    def items(self):
        return self.__dict__.items()
    
    ### VALUES
    def values(self):
        return self.__dict__.values()
    
    ### RENAME CHANNELS
    def rename_channels(self, mapping):
        """
        Renombra los canales existentes según el diccionario dado.
        Si un canal no existe en ch_names, lo informa pero no detiene la ejecución.
        Si el renombrado genera canales duplicados, no se aplica ningún cambio.

        Parameters
        ----------
        mapping : dict
            Diccionario con pares {nombre_antiguo: nombre_nuevo}.
        """
        if not self.ch_names:
            print("No hay nombres de canales definidos.")
            return

        not_found = []
        new_ch_names = []

        for ch in self.ch_names:
            if ch in mapping:
                new_ch_names.append(mapping[ch])
            else:
                new_ch_names.append(ch)

        # Verificar canales no encontrados
        for old_name in mapping:
            if old_name not in self.ch_names:
                not_found.append(old_name)

        # Verificar duplicados
        if len(set(new_ch_names)) != len(new_ch_names):
            print("Error: el renombrado genera canales duplicados. No se aplicaron cambios.")
            return

        # Asignar si todo está bien
        self.ch_names = new_ch_names

        if not_found:
            print(f"Aviso: los siguientes canales no existen y no fueron renombrados: {not_found}")
        else:
            print(f"Canales renombrados exitosamente. Los canales ahora son: {self.ch_names}")

    
    def pretty(self):
        """Devuelve las claves y valores en formato de tabla bonita (pandas)."""
        return pd.DataFrame(self.__dict__.items(), columns=["Clave", "Valor"])
