
from RawSignal import RawSignal
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.time_frequency import tfr_array_morlet

class EMGSignal:
    """
    Clase para el procesamiento y visualización de señales EMG.

    Esta clase permite cargar señales EMG desde un objeto RawSignal, detectar activaciones
    musculares mediante un umbral, y visualizar la señal cruda, el umbral, transformadas
    de frecuencia y envolventes mediante la transformada de Hilbert.

    Parámetros:
    raw_signal : RawSignal
        Objeto que contiene la señal EMG y su información asociada.
    umbral : float, opcional
        Valor umbral para detección de activaciones (por defecto 200).
    canal : int, str, list o None, opcional
        Canal o canales a utilizar.
    
    Atributos:
    signal : RawSignal
        Objeto de señal original.
    umbral : float
        Valor umbral para detección de activaciones.
    fm : float
        Frecuencia de muestreo de la señal.
    indices_temporales : list of np.array
        Índices temporales donde se detectan activaciones.
    """

    def __init__(self, signal: RawSignal, umbral=20):
        """
        Clase para análisis de señales EMG.

        Parámetros:
        -----------
        signal : RawSignal
            Objeto RawSignal con los datos EMG y su información de canal.
        umbral : float, opcional
            Valor de umbral para detección de activaciones.
        """
        if not isinstance(signal, RawSignal):
            raise TypeError("La señal debe ser una instancia de RawSignal")

        if signal.data.ndim != 2:
            raise ValueError("Los datos de la señal deben tener shape (n_canales, n_muestras)")

        self.signal = signal
        self.umbral = umbral
        self.sfreq = signal.sfreq
        self.indices_temporales = []

        # Crear objeto Raw de MNE con ch_types como lista si hay múltiples canales
        ch_types = ['misc'] * signal.data.shape[0]
        self._raw_mne = mne.io.RawArray(
            self.signal.data.copy(),
            mne.create_info(self.signal.info.ch_names, self.signal.info.sfreq, ch_types=ch_types)
        )

    def detectar_activaciones(self):
        """
        Detecta activaciones en cada canal comparando el valor absoluto de la señal con el umbral.

        Retorna:
        list of np.ndarray
            Lista con arrays de índices donde se superó el umbral en cada canal.
        """
        self.indices_temporales = []
        for canal in self.signal.data:
            indices = np.where(np.abs(canal) > self.umbral)[0]
            self.indices_temporales.append(indices)

        return self.indices_temporales

    def graficar_emg_con_umbral(self, t_inicio=0, t_fin=10, mostrar_onset=False):
        """
        Grafica la señal EMG con umbral y activaciones detectadas.

        Parámetros:
        t_inicio : float, opcional
            Tiempo inicial en segundos.
        t_fin : float, opcional
            Tiempo final en segundos.
        mostrar_onset : bool, opcional
            Si es True, grafica marcas de onset de anotaciones.

        Raises:
        ValueError
            Si los tiempos son inválidos.
        """
        if t_inicio < 0 or t_fin <= t_inicio:
            raise ValueError("t_inicio debe ser >= 0 y t_fin > t_inicio")

        inicio = int(t_inicio * self.sfreq)
        fin = int(t_fin * self.sfreq)

        # Ajustar fin si excede el largo de la señal
        if fin > self.signal.data.shape[1]:
            fin = self.signal.data.shape[1]

        n_canales = self.signal.data.shape[0]
        tiempo = np.arange(inicio, fin) / self.sfreq

        # Obtener onset filtrados si se solicita
        onset_filtrado = []
        if mostrar_onset and hasattr(self.signal, 'anotaciones') and self.signal.anotaciones is not None:
            df_anot = self.signal.anotaciones.anotations
            onset_filtrado = df_anot.loc[
                (df_anot["onset"] >= t_inicio) & (df_anot["onset"] <= t_fin), "onset"
            ].values

        plt.figure(figsize=(12, 4 * n_canales))

        for i in range(n_canales):
            señal_completa = self.signal.data[i]
            señal = señal_completa[inicio:fin]
            indices_activados = np.where(np.abs(señal) > self.umbral)[0]

            plt.subplot(n_canales, 1, i + 1)
            plt.plot(tiempo, señal, label='Señal EMG', color='#c2185b')
            plt.axhline(self.umbral, color='red', linestyle='--', label=f'Umbral = {self.umbral}')
            plt.axhline(-self.umbral, color='red', linestyle='--')
            plt.scatter(tiempo[indices_activados], señal[indices_activados],
                        color='orange', label='Activaciones', zorder=3)

            if mostrar_onset and len(onset_filtrado) > 0:
                for t in onset_filtrado:
                    plt.axvline(t, color='red', linestyle='--', alpha=0.7)
                    # Ubico el marcador en el punto máximo absoluto + 10% margen para visibilidad
                    max_val = np.max(np.abs(señal))
                    plt.plot(t, max_val * 1.1, marker='v', color='red')

            plt.xlabel('Tiempo [s]')
            plt.ylabel('Amplitud (uV)')
            plt.title(f'Señal EMG - Canal {i} ({t_inicio} a {t_fin} s)', fontweight='bold')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.show()

    def calcular_tfr(self, canal, frecs=np.arange(2, 40, 2), n_cycles=5, use_log=True, plot=True):
        """
        Calcula la representación tiempo-frecuencia (wavelets Morlet) para un canal de EMG.

        Parámetros:
        canal : str o int
            Nombre o índice del canal.
        frecs : array-like
            Vector de frecuencias para la transformada.
        n_cycles : int o array-like
            Número de ciclos de la wavelet.
        use_log : bool
            Si es True, la potencia se muestra en escala logarítmica.
        plot : bool
            Si es True, se grafica la TFR.

        Retorna:
        power : np.ndarray
            Matriz potencia tiempo-frecuencia.
        times : np.ndarray
            Vector de tiempos.
        """
        # Obtener índice del canal
        if isinstance(canal, str):
            if canal not in self.signal.info.ch_names:
                raise ValueError(f"Canal '{canal}' no existe en la señal.")
            idx = self.signal.info.ch_names.index(canal)
        elif isinstance(canal, int):
            if canal < 0 or canal >= len(self.signal.info.ch_names):
                raise ValueError(f"Índice de canal fuera de rango (0 - {len(self.signal.info.ch_names)-1}).")
            idx = canal
        else:
            raise TypeError("El parámetro 'canal' debe ser string o entero.")

        # Obtener datos del canal para la TFR
        data = self._raw_mne.get_data(picks=[idx])[np.newaxis, :, :]
        sfreq = self.sfreq

        power = tfr_array_morlet(data, sfreq=sfreq, freqs=frecs, n_cycles=n_cycles, output='power')[0, 0]
        times = self._raw_mne.times

        if plot:
            plt.figure(figsize=(10, 4))
            plt.imshow(np.log10(power) if use_log else power,
                       aspect='auto', origin='lower',
                       extent=[times[0], times[-1], frecs[0], frecs[-1]],
                       cmap='magma')
            plt.colorbar(label='Potencia' + (' (log)' if use_log else ''))
            plt.xlabel("Tiempo (s)")
            plt.ylabel("Frecuencia (Hz)")
            plt.title(f"TFR - Canal {canal}")
            plt.tight_layout()
            plt.show()

        return power, times    
    
    def calcular_hilbert(self, canales=None, envelope=False, return_magnitude_and_phase=False,
                     recorte_segundos=(0, 0), clip_percentiles=None, plot=True):
        """
        Calcula la transformada de Hilbert y permite devolver magnitud, fase o señal compleja.

        Parámetros:
        canales : str, int, list de str/int o None
            Canales sobre los que calcular la transformada. None para todos.
        envelope : bool
            Si es True, devuelve la envolvente (magnitud).
        return_magnitude_and_phase : bool
            Si es True, devuelve tupla (magnitud, fase).
        recorte_segundos : tuple (ini, fin)
            Tiempo a recortar al inicio y final.
        clip_percentiles : tuple o None
            Percentiles para recorte en la gráfica.
        plot : bool
            Si es True y un solo canal, grafica el resultado.

        Retorna:
        np.ndarray o tuple de np.ndarray
            Resultado de la transformada.
        """
        raw_copy = self._raw_mne.copy()

        # Obtener nombres de canales si existen
        ch_names = self.signal.info.ch_names
        if ch_names is None or len(ch_names) == 0:
            n_ch = self.signal.data.shape[0]
            ch_names = [f"canal_{i}" for i in range(n_ch)]

        if canales is None:
            picks = list(range(len(ch_names)))
        elif isinstance(canales, (str, int)):
            if isinstance(canales, int):
                if canales < 0 or canales >= len(ch_names):
                    raise ValueError(f"Índice de canal fuera de rango (0 - {len(ch_names)-1}).")
                picks = [canales]
            else:
                if canales not in ch_names:
                    raise ValueError(f"Canal '{canales}' no existe en la señal.")
                # Convertir a índice
                picks = [ch_names.index(canales)]
        elif isinstance(canales, list):
            picks = []
            for ch in canales:
                if isinstance(ch, int):
                    if ch < 0 or ch >= len(ch_names):
                        raise ValueError(f"Índice de canal fuera de rango (0 - {len(ch_names)-1}).")
                    picks.append(ch)
                elif isinstance(ch, str):
                    if ch not in ch_names:
                        raise ValueError(f"Canal '{ch}' no existe en la señal.")
                    picks.append(ch_names.index(ch))
                else:
                    raise TypeError("Elementos de 'canales' deben ser str o int.")
        else:
            raise TypeError("El argumento 'canales' debe ser str, int, list[str|int] o None.")

        # Aplicar transformada de Hilbert con picks como índices enteros
        raw_hilbert = raw_copy.apply_hilbert(picks=picks, envelope=envelope)
        hilbert_data = raw_hilbert.get_data(picks=picks)

        # Definir salida
        if envelope:
            resultado = hilbert_data
        elif return_magnitude_and_phase:
            magnitud = np.abs(hilbert_data)
            fase = np.angle(hilbert_data)
            resultado = (magnitud, fase)
        else:
            resultado = hilbert_data

        # Recorte temporal
        t = raw_copy.times
        srate = raw_copy.info['sfreq']
        i_ini = int(recorte_segundos[0] * srate)
        i_fin = int(t.shape[0] - recorte_segundos[1] * srate)
        t = t[i_ini:i_fin]

        if isinstance(resultado, tuple):
            resultado = tuple(r[:, i_ini:i_fin] for r in resultado)
        else:
            resultado = resultado[:, i_ini:i_fin]

        # Gráfica si corresponde
        n_canales = resultado[0].shape[0] if isinstance(resultado, tuple) else resultado.shape[0]
        if plot and n_canales == 1:
            y = resultado[0][0] if isinstance(resultado, tuple) else resultado[0]

            ymin, ymax = (np.percentile(y, clip_percentiles[0]), np.percentile(y, clip_percentiles[1])) if clip_percentiles else (np.min(y), np.max(y))

            plt.figure(figsize=(10, 4))
            plt.plot(t, y, color='mediumvioletred')
            canal_nombre = ch_names[picks[0]] if picks else "Canal desconocido"
            plt.title(f"Transformada de Hilbert - Canal {canal_nombre}")
            plt.xlabel("Tiempo (s)")
            plt.ylabel("Amplitud" if envelope else "Real (Hilbert)")
            plt.ylim(ymin, ymax)
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.show()