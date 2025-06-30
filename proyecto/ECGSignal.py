
import numpy as np
import matplotlib.pyplot as plt
from RawSignal import RawSignal
from mne.time_frequency import tfr_array_morlet
from scipy.signal import find_peaks
import mne

class ECGSignal:
    """
    Clase para el procesamiento y análisis de señales ECG.

    Permite seleccionar canales específicos, detectar los picos R de las señales ECG,
    calcular la frecuencia cardíaca y visualizar los resultados mediante gráficas
    de señal en el tiempo y espectrogramas.

    Atributos:
    signal : RawSignal
        Objeto con la señal bruta y la información de los canales.
    sfreq : float
        Frecuencia de muestreo de la señal (Hz).
    picos_r : list of np.ndarray
        Índices de los picos R detectados por canal.
    frec_cardiaca : list of float
        Frecuencias cardíacas estimadas para cada canal.
    """
    def __init__(self, signal: RawSignal, umbral=0.5):
        """
        Clase para análisis de señales ECG.

        Parámetros:
        signal : RawSignal
            Objeto RawSignal con los datos ECG y su información de canal.
        umbral : float, opcional
            Valor de umbral para detección de picos R (en amplitud, ejemplo 0.5).
        """
        if not isinstance(signal, RawSignal):
            raise TypeError("La señal debe ser una instancia de RawSignal")

        if signal.data.ndim != 2:
            raise ValueError("Los datos de la señal deben tener shape (n_canales, n_muestras)")

        self.signal = signal
        self.umbral = umbral
        self.sfreq = signal.sfreq
        self.picos_r = []  # Para guardar índices de picos R detectados

        # Crear objeto Raw de MNE con ch_types 'misc' para todos los canales
        ch_types = ['misc'] * signal.data.shape[0]
        # Crear ch_names si no existen
        if (self.signal.info.ch_names is None) or (len(self.signal.info.ch_names) == 0):
            ch_names = [f"ECG{i}" for i in range(self.signal.data.shape[0])]
        else:
            ch_names = self.signal.info.ch_names

        self._raw_mne = mne.io.RawArray(
            self.signal.data.copy(),
            mne.create_info(ch_names, self.signal.info.sfreq, ch_types=ch_types)
        )

    def detectar_picos_R(self, distancia_minima=None, height=None):
        """
        Detecta los picos R en cada canal de la señal ECG utilizando un umbral de altura.

        Parámetros:
        distancia_minima : float, opcional
            Distancia mínima entre picos, en segundos. Si se especifica, se convierte a muestras.
        height : float, opcional
            Altura mínima para considerar un pico. Si es None, se calcula automáticamente como
            media + 3 * desviación estándar.

        Retorna:
        list of np.ndarray
            Lista con arrays de los índices de los picos R detectados por canal.
        """
        if distancia_minima is not None:
            distancia_minima = int(self.sfreq * distancia_minima)

        self.picos_r = []  # reinicio por si se llama varias veces

        for canal in self.signal.data:
            if height is None:
                media = np.mean(canal)
                std = np.std(canal)
                height_auto = media + 3 * std
            else:
                height_auto = height

            peaks, _ = find_peaks(canal, distance=distancia_minima, height=height_auto)
            self.picos_r.append(peaks)

        return self.picos_r

    def calcular_frecuencia_cardiaca(self):
        """
        Calcula la frecuencia cardíaca media a partir de los intervalos RR de los picos R detectados.

        Retorna:
        list of float
            Lista con la frecuencia cardíaca (en latidos por minuto) para cada canal.
        """
        if not self.picos_r:
            self.detectar_picos_R()
        
        self.frec_cardiaca = []
        for peaks in self.picos_r:
            if len(peaks) < 2:
                # No hay suficientes picos para calcular intervalos RR
                self.frec_cardiaca.append(0)
                continue
            rr_intervals = np.diff(peaks) / self.sfreq
            mean_rr = np.mean(rr_intervals)
            frecuencia = 60 / mean_rr if mean_rr > 0 else 0
            self.frec_cardiaca.append(frecuencia)

        return self.frec_cardiaca

    def graficar_señal_con_picos(self, t_inicio=0, t_fin=10):
        """
        Grafica la señal ECG entre los tiempos indicados (en segundos), junto con los picos R detectados.

        Parámetros:
        t_inicio : float, opcional
            Tiempo inicial en segundos (por defecto 0).
        t_fin : float, opcional
            Tiempo final en segundos (por defecto 10).
        """
        inicio = int(t_inicio * self.sfreq)
        fin = int(t_fin * self.sfreq)

        if not self.picos_r:
            self.detectar_picos_R()

        tiempo = np.arange(inicio, fin) / self.sfreq

        n_canales = self.signal.data.shape[0]

        for i in range(n_canales):
            signal = self.signal.data[i]
            picos = self.picos_r[i]
            # Filtrar picos visibles en el rango seleccionado
            picos_visibles = picos[(picos >= inicio) & (picos < fin)]

            plt.figure(figsize=(10, 4))
            plt.plot(tiempo, signal[inicio:fin], label="Señal ECG", color="#c2185b")
            plt.scatter(picos_visibles / self.sfreq, signal[picos_visibles], 
                        color='#f8f990', edgecolors='#880e4f', s=70, label="Picos R")
            plt.title(f"Canal {i}", fontweight="bold", color="#880e4f")
            plt.xlabel("Tiempo [s]", color="#880e4f")
            plt.ylabel("Amplitud", color="#880e4f")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    def calcular_tfr(self, canal_idx=0, frecs=np.arange(2, 40, 2), n_cycles=5, use_log=True, plot=True):
        """
        Calcula la representación tiempo-frecuencia (wavelets Morlet) para un canal ECG.

        Parámetros:
        canal_idx : int
            Índice del canal a analizar.
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
        n_canales = self.signal.data.shape[0]
        if not (0 <= canal_idx < n_canales):
            raise ValueError(f"Índice de canal fuera de rango (0 - {n_canales - 1}).")

        data = self._raw_mne.get_data(picks=[canal_idx])[np.newaxis, :, :]
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
            plt.title(f"TFR - Canal {canal_idx}")
            plt.tight_layout()
            plt.show()

        return power, times