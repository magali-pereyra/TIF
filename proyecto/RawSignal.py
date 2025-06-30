import numpy as np
import pandas as pd
from Info import Info
from Anotations import Anotaciones
import mne as mne
from mne.filter import filter_data, notch_filter 

class RawSignal:
    def __init__(self, data: np.ndarray, sfreq: float, info,  # info es de tipo Info
                first_samp: int = 0, anotaciones=None):      # anotaciones es de tipo Anotaciones
        """
        Clase para almacenar señales fisiológicas crudas.

        Parámetros:
        data : np.ndarray
            Matriz de datos con forma (n_canales, n_muestras).
        sfreq : float
            Frecuencia de muestreo de la señal en Hz.
        info : Info
            Objeto que contiene metadatos de la señal.
        first_samp : int, opcional
            Índice de la primera muestra válida o a tener en cuenta.
        anotaciones : Anotaciones, opcional
            Eventos o marcas asociadas a la señal.

        Raises:
        ValueError
            Si el array 'data' no tiene la forma '(n_canales, n_muestras)'.
        ValueError
            Si el número de canales en 'data' no coincide con la cantidad de nombres en 'info.ch_names'.
        ValueError
            Si la frecuencia de muestreo 'sfreq' no coincide con 'info.sfreq'.
        ValueError
            Si el índice 'first_samp' está fuera del rango de la señal.
        """
        if data.ndim != 2:
            raise ValueError("El array 'data' debe tener forma (n_canales, n_muestras).")

        if info.ch_names is not None and data.shape[0] != len(info.ch_names):
            raise ValueError("El número de canales en 'data' no coincide con 'info.ch_names'.")

        if not np.isclose(sfreq, info.sfreq):
            raise ValueError("La frecuencia de muestreo en 'sfreq' y 'info.sfreq' debe coincidir.")

        if not (0 <= first_samp < data.shape[1]):
            raise ValueError("El índice 'first_samp' está fuera del rango de la señal.")

        self.data = data
        self.sfreq = sfreq
        self.info = info
        self.first_samp = first_samp
        self.anotaciones = anotaciones

    def __repr__(self):
        """
        Devuelve una representación en texto del objeto RawSignal.

        Esta representación incluye el número de canales, la cantidad total de muestras,
        la frecuencia de muestreo y el índice de la primera muestra válida.

        Retorna:

        str:
            Una cadena con información resumida del objeto RawSignal.
        """
        return (f"<RawSignal | {self.data.shape[0]} canales × {self.data.shape[1]} muestras, "
                f"sfreq={self.sfreq} Hz, first_samp={self.first_samp}>")
        
    def get_data(self, picks=None, start=0, stop=None, reject=None, times=False):
        """
        Extrae un segmento de datos del objeto, con opción de excluir canales y aplicar umbral de rechazo.

        Parámetros:
        picks : list de str o str, opcional
            Canal o lista de canales a extraer. Si es None, se devuelven todos los canales.
        start : float, opcional
            Tiempo inicial del segmento en segundos. Valor por defecto: 0.
        stop : float, opcional
            Tiempo final del segmento en segundos. Si es None, se usa el final de la señal.
        reject : float, opcional
            Umbral para filtrar canales cuyo valor absoluto máximo exceda este valor.
            Si es None, se usa self.reject si está definido.
        times : bool, opcional
            Si es True, también devuelve un vector de tiempos correspondiente.

        Retorna:
        np.ndarray
            Matriz con los datos seleccionados (n_canales x n_muestras).
        np.ndarray (opcional)
            Vector de tiempos (solo si 'times=True').
        """
        data_copy = self.data.copy()

        if isinstance(picks, str):
            # picks es un solo canal (string)
            if picks in self.info.ch_names:
                indices = [self.info.ch_names.index(picks)]
            else:
                raise ValueError(f"No existe el canal {picks}")

        elif isinstance(picks, (list, tuple)):
            # picks es una lista o tupla de canales
            indices = []
            for ch in picks:
                if ch in self.info.ch_names:
                    indices.append(self.info.ch_names.index(ch))
                else:
                    raise ValueError(f"No existe el canal {ch}")

        # Exclusión de canales
        if picks is not None:
                ch_selected=  data_copy[indices,:]
        else:
                ch_selected= data_copy

        # Conversión de tiempo a muestras
        m_inicio = int(start * self.sfreq)
        m_fin = int(stop * self.sfreq) if stop is not None else ch_selected.shape[1]

        if m_inicio >= m_fin or m_fin > ch_selected.shape[1]:
            raise ValueError("Rango de tiempo inválido.")

        segmento = ch_selected[:, m_inicio:m_fin]

        if reject is not None:
            max_abs = np.max(np.abs(segmento), axis=1)
            canales_validos = max_abs < reject
            segmento = segmento[canales_validos, :]

    # Devolver datos y tiempo (si se pide)
        if times:
            t = np.arange(m_inicio, m_fin) / self.sfreq
            return segmento, t
        else:
            return segmento
    
    def drop_channels(self, ch):
        """
        Devuelve una nueva instancia de RawSignal sin los canales especificados.

        Parámetros:
        ch : str o list de str
            Canal o lista de canales a eliminar.

        Retorna:
        RawSignal
            Nueva instancia sin los canales eliminados.
        """
        if isinstance(ch, str):
            ch = [ch]

        # Índices a eliminar
        indices_to_remove = []
        for canal in ch:
            if canal in self.info.ch_names:
                indices_to_remove.append(self.info.ch_names.index(canal))
            else:
                raise ValueError(f"Canal {canal} no encontrado en info.ch_names")

        # Índices que queremos conservar
        indices_total = list(range(self.data.shape[0]))
        indices_restantes = [i for i in indices_total if i not in indices_to_remove]

        # Filtrar datos
        data_filtrada = self.data[indices_restantes, :]

        # Filtrar info
        ch_names_nuevos = [self.info.ch_names[i] for i in indices_restantes]
        ch_types_nuevos = [self.info.ch_types[i] for i in indices_restantes] if isinstance(self.info.ch_types, list) else self.info.ch_types
        bads_nuevos = [c for c in self.info.bads if c not in ch]

        info_nuevo = Info(
            ch_names=ch_names_nuevos,
            ch_types=ch_types_nuevos,
            sfreq=self.sfreq,
            description=self.info.description,
            experimenter=self.info.experimenter,
            subject_info=self.info.subject_info,
            bads=bads_nuevos
        )

        # Devolver nuevo objeto RawSignal
        return RawSignal(
            data=data_filtrada,
            sfreq=self.sfreq,
            info=info_nuevo,
            first_samp=self.first_samp,
            anotaciones=self.anotaciones
        )
    
    def crop(self,tin=0.0, tf=None):
        """
        Recorta la señal en un intervalo de tiempo especificado y ajusta las anotaciones en ese rango.

        Este método genera una nueva instancia de RawSignal con los datos comprendidos entre los 
        tiempos 'tin' y 'tf' (en segundos), y actualiza las anotaciones para que coincidan con el nuevo
        intervalo temporal. Los eventos fuera del rango son descartados, y los 'onset' de los eventos 
        restantes se reajustan para que comiencen desde cero.

        Parameters:
        tin : float, optional
            Tiempo inicial del recorte en segundos (por defecto 0.0).
        tf : float, optional
            Tiempo final del recorte en segundos. Si es None, se usa hasta el final de la señal.

        Retorna:
        RawSignal
            Nueva instancia de RawSignal con los datos y anotaciones recortados.

        Raises:
        ValueError
            Si los valores de 'tin' y 'tf' están fuera del rango de la señal.
        """
        # Recortar datos
        data_cortada = self.get_data(picks=self.info.ch_names, start=tin, stop=tf, times=False)

        nuevas_anotaciones = None
        df_anot = self.anotaciones.anotations.copy()

        if tf is not None:
            eventos_filtrados = df_anot[(df_anot['onset'] >= tin) & (df_anot['onset'] <= tf)].copy()
        else:
            eventos_filtrados = df_anot[df_anot['onset'] >= tin].copy()

            # Reajustar onsets al nuevo inicio
        eventos_filtrados['onset'] -= tin

            # Crear nueva instancia de Anotaciones
        nuevas_anotaciones = Anotaciones(
            onset=eventos_filtrados['onset'].tolist(),
            duration=eventos_filtrados['duration'].tolist(),
            description=eventos_filtrados['event_id'].tolist()
            )

        # Devolver nuevo objeto RawSignal
        return RawSignal(
            data=data_cortada,
            sfreq=self.sfreq,
            info=self.info,
            first_samp=0,
            anotaciones=nuevas_anotaciones
        )
    
    def describe(self):
        """
        Genera un resumen estadístico básico de cada canal de la señal.

        Para cada canal, calcula el valor mínimo, primer cuartil (Q1), mediana,
        tercer cuartil (Q3) y valor máximo. Además, incluye el nombre y tipo
        de canal si están disponibles en el objeto Info.

        Retorna:
        pandas.DataFrame
            DataFrame con una fila por canal y las siguientes columnas:
            - 'name'     : Nombre del canal
            - 'type'     : Tipo de canal (e.g. 'EEG', 'ECG', etc.)
            - 'min'      : Valor mínimo del canal
            - 'Q1'       : Primer cuartil (25%)
            - 'mediana'  : Mediana (50%)
            - 'Q3'       : Tercer cuartil (75%)
            - 'max'      : Valor máximo del canal
        """
        resumen = []

        for idx, canal in enumerate(self.data):
            nombre = self.info.ch_names[idx] if self.info.ch_names else f"Canal {idx}"
            tipo = self.info.ch_types[idx] if hasattr(self.info, 'ch_types') and self.info.ch_types else "desconocido"

            canal_stats = {
                "name": nombre,
                "type": tipo,
                "min": np.min(canal),
                "Q1": np.percentile(canal, 25),
                "mediana": np.median(canal),
                "Q3": np.percentile(canal, 75),
                "max": np.max(canal)
            }
            resumen.append(canal_stats)

        return pd.DataFrame(resumen)

    def filter(self, l_freq: float, h_freq: float, notch_freq: float = 50., order: int = 4, fir_window: str = "hamming"):
        """
        Aplica un filtro pasabanda y notch a la señal.

        Parametros:
        l_freq : float
            Frecuencia de corte baja (Hz) para el filtro pasabanda.
        h_freq : float
            Frecuencia de corte alta (Hz) para el filtro pasabanda.
        notch_freq : float , optional
            Frecuencia del filtro notch para eliminar ruido (por defecto 50 Hz).
        order : int, optional
            Orden del filtro (por defecto 4).
        fir_window : str, optional
            Tipo de ventana para el diseño del filtro FIR (por defecto "hamming").

        Retorna:
        RawSignal
            Nueva instancia de 'RawSignal' con los datos filtrados.
            
        Raises:
        ValueError
            Si los valores de 'l_freq' o 'h_freq' no son válidos.
        ValueError
            Si el valor de 'notch_freq' no es positivo.
        """
        if l_freq is None or h_freq is None or l_freq <= 0 or h_freq <= 0 or l_freq >= h_freq:
            raise ValueError("Los valores de 'l_freq' y 'h_freq' deben ser positivos y l_freq < h_freq.")

        if notch_freq is not None and notch_freq <= 0:
            raise ValueError("El valor de 'notch_freq' debe ser positivo.")

        # Copia de datos para no modificar el original
        data_filt = self.data.copy()
        # Asegurar tipo float64 para los filtros de MNE
        data_filt = data_filt.astype(np.float64)

        # Filtro notch
        if notch_freq is not None:
            data_filt = notch_filter(data_filt, self.sfreq, freqs=notch_freq, method='iir')

        # Filtro pasabanda
        data_filt = filter_data(
            data_filt,
            sfreq=self.sfreq,
            l_freq=l_freq,
            h_freq=h_freq,
            method='iir',
            verbose=False
        )

        # Devolver nueva instancia de RawSignal con los datos filtrados
        return RawSignal(data_filt, self.sfreq, self.info, self.first_samp, self.anotaciones)
    
    def cut_before_event(self, margen_segundos=5.0):
        """
        Recorta la señal para que comience 'margen_segundos' antes del primer evento registrado en las anotaciones.

        Parámetros:
        margen_segundos : float
            Tiempo en segundos a restar desde el primer evento.

        Retorna:
        RawSignal
            Nueva instancia recortada de RawSignal.

        Raises:
            ValueError
            Si no hay anotaciones cargadas o no contienen la columna 'onset'.
            Si el tiempo de inicio recortado es negativo.
        """
        if self.anotaciones is None or not hasattr(self.anotaciones, 'anotations'):
            raise ValueError("No hay anotaciones cargadas en el objeto.")
        
        eventos = self.anotaciones.anotations

        if 'onset' not in eventos.columns:
            raise ValueError("Las anotaciones no contienen una columna 'onset'.")

        primer_onset = eventos['onset'].min()
        inicio_seg = primer_onset - margen_segundos

        if inicio_seg < 0:
            raise ValueError(f"El margen de {margen_segundos} segundos hace que el inicio ({inicio_seg}s) sea negativo.")

        inicio_muestra = int(inicio_seg * self.sfreq)

        datos_recortados = self.data[:, inicio_muestra:]

        return RawSignal(
            data=datos_recortados,
            sfreq=self.sfreq,
            info=self.info,
            first_samp=inicio_muestra,
            anotaciones=self.anotaciones
        )
    
    def pick(self, picks):
        """
        Crea una nueva instancia de RawSignal que contiene solo los canales seleccionados.

        Este método permite seleccionar uno o más canales específicos por nombre o índice,
        y devuelve una copia del objeto RawSignal con esos canales. Se actualiza también
        el objeto Info correspondiente.

        Parametros:
        picks : str, int o list de str/int
            Nombre(s) o índice(s) de los canales a seleccionar.

        Retorna:
        RawSignal
            Nueva instancia de RawSignal con los canales seleccionados.

        Raises:
        TypeError
            Si el argumento no es un str, int o lista de ellos.
        ValueError
            Si algún nombre o índice de canal no es válido.
        """
        if isinstance(picks, (str, int)):
            picks = [picks]
        elif isinstance(picks, list):
            if not all(isinstance(p, (str, int)) for p in picks):
                raise ValueError("Todos los elementos deben ser str o int.")
        else:
            raise TypeError("El argumento 'picks' debe ser str, int o lista de ellos.")

        # Obtener nombres y tipos de los canales seleccionados
        picks_names = []
        picks_types = []

        for p in picks:
            if isinstance(p, int):
                picks_names.append(self.info.ch_names[p])
                picks_types.append(self.info.ch_types[p])
            elif isinstance(p, str):
                if p in self.info.ch_names:
                    i = self.info.ch_names.index(p)
                    picks_names.append(p)
                    picks_types.append(self.info.ch_types[i])
                else:
                    raise ValueError(f"Canal '{p}' no encontrado.")

        # Obtener los datos seleccionados
        new_data = self.get_data(picks=picks_names)

        # Crear nuevo objeto Info con tipos correctos
        new_info = Info(ch_names=picks_names, ch_types=picks_types, sfreq=self.sfreq)

        # Devolver nuevo RawSignal
        return RawSignal(data=new_data, sfreq=self.sfreq, info=new_info, anotaciones=self.anotaciones)

        
    def __getitem__(self, key):
        """
        Permite acceder a segmentos de datos de la señal utilizando una notación tipo array.

        Este método sobrecarga el operador [] para acceder a datos por canal (o canales)
        y por rango temporal de muestras, devolviendo tanto los datos como el vector
        de tiempos correspondiente.

        Parámetros:
        key : str, tuple o list
        
        Retorna:
        tuple
            - datos : np.ndarray
                Matriz de datos (n_canales x n_muestras) correspondiente al canal o canales seleccionados.
            - tiempo : np.ndarray
                Vector de tiempo asociado (en segundos) para el rango de muestras devuelto.

        Raises:
        TypeError
            Si el formato de acceso no es válido.
        ValueError
            Si se especifica un canal inexistente o si el rango de muestras está fuera de los límites.
        """
        if isinstance(key, str):
            picks = [key]
            s = slice(None)
        elif all(isinstance(pick, str) for pick in key):
            picks = key
            s = slice(None)
        elif isinstance(key, tuple) and len(key) == 2:
            picks, s = key
            if isinstance(picks, str):
                picks = [picks]
        else:
            raise TypeError("Formato no soportado. Use 'canal', ('canal', slice) o ('lista', slice).")

        # Convertir picks a índices
        indices = []
        for canal in picks:
            if canal in self.info.ch_names:
                indices.append(self.info.ch_names.index(canal))
            else:
                raise ValueError(f"Canal '{canal}' no encontrado.")

        # Asegurar slice válido
        start = s.start if s.start is not None else 0
        stop = s.stop if s.stop is not None else self.data.shape[1]
        step = s.step if s.step is not None else 1

        # Verificar límites
        if not (0 <= start < self.data.shape[1]) or not (0 < stop <= self.data.shape[1]):
            raise ValueError("Índices fuera de rango.")

        # Extraer datos
        datos = self.data[indices, start:stop:step]

        # Generar vector de tiempos
        tiempo = np.arange(start, stop, step) / self.sfreq

        return datos, tiempo

    def plot(self,start=0,scalings=40,color="crimson"):
        """
        Grafica la señal fisiológica utilizando la herramienta de visualización de MNE.

        Este método convierte la señal actual en un objeto Raw de MNE y muestra
        una interfaz interactiva para explorar los datos por canal.

        Parametros:
        start : float, optional
            Tiempo inicial (en segundos) desde donde comenzar la visualización (por defecto 0).
        scalings : float, optional
            Escala vertical de la señal para la visualización, en µV por unidad (por defecto 40).

        Retorna:
        matplotlib.figure.Figure
            Objeto de figura generado por MNE para su visualización.

        Notas:
        - Requiere "mne".
        """
        if self.info.ch_names is None:
            self.info.ch_names=["Canal no definido"]
        objeto_raw=mne.io.RawArray(self.data,mne.create_info(self.info.ch_names,self.info.sfreq))
        return objeto_raw.plot(start=start,scalings=scalings,color=color)
    
    def set_anotaciones(self, anotaciones_nuevas):
        """
        Asocia un objeto de tipo 'Anotaciones' a la señal fisiológica.

        Parametros:
        anotaciones_nuevas : Anotaciones
            Objeto de la clase 'Anotaciones' que contiene los eventos.

        Raises:
        TypeError
            Si 'anotaciones_nuevas' no es una instancia de la clase 'Anotaciones'.
        ValueError
            Si alguna anotación tiene un 'onset' fuera del rango de la señal.
        """

        # Verificar tipo
        if not isinstance(anotaciones_nuevas, Anotaciones):
            raise TypeError("El parámetro debe ser una instancia de la clase 'Anotaciones'.")

        duracion_total = self.data.shape[1] / self.sfreq  # duración en segundos

        onsets = anotaciones_nuevas.anotations['onset']
        if (onsets < 0).any() or (onsets > duracion_total).any():
            raise ValueError("Hay anotaciones fuera del rango de duración de la señal.")

        # Asignar si todo está bien
        self.anotaciones = anotaciones_nuevas

    def remove_segment(self, t_start, t_stop):
        """
        Elimina un segmento de la señal entre t_start y t_stop (en segundos) 
        y ajusta las anotaciones en consecuencia.

        Parámetros:
        t_start : float
            Tiempo de inicio del segmento a eliminar (en segundos).
        t_stop : float
            Tiempo de fin del segmento a eliminar (en segundos).

        Retorna:
        RawSignal
            Nueva instancia de RawSignal con el segmento eliminado y anotaciones ajustadas.

        Raises:
        ValueError
            Si los tiempos están fuera de rango o t_start >= t_stop.
        """
        if t_start >= t_stop:
            raise ValueError("t_start debe ser menor que t_stop.")

        total_duration = self.data.shape[1] / self.sfreq
        if t_start < 0 or t_stop > total_duration:
            raise ValueError("Intervalo fuera del rango de la señal.")

        # Convertir a muestras
        m_start = int(np.round(t_start * self.sfreq))
        m_stop = int(np.round(t_stop * self.sfreq))
        delta_m = m_stop - m_start
        delta_t = t_stop - t_start

        # Eliminar segmento de la señal
        data_new = np.concatenate((self.data[:, :m_start], self.data[:, m_stop:]), axis=1)

        # Ajustar anotaciones
        nuevas_anotaciones = None
        if self.anotaciones is not None:
            df = self.anotaciones.anotations.copy()

            # Mantener solo anotaciones que están fuera del segmento
            df_filtrado = df[~((df["onset"] >= t_start) & (df["onset"] < t_stop))].copy()

            # Ajustar anotaciones posteriores al segmento
            df_filtrado.loc[df_filtrado["onset"] >= t_stop, "onset"] -= delta_t

            nuevas_anotaciones = Anotaciones(
                onset=df_filtrado["onset"].tolist(),
                duration=df_filtrado["duration"].tolist(),
                description=df_filtrado["event_id"].tolist()
            )

        # Crear nuevo objeto RawSignal
        return RawSignal(data=data_new, sfreq=self.sfreq, info=self.info, first_samp=0, anotaciones=nuevas_anotaciones)