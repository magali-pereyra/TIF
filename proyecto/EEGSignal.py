# importo las librerías necesarias para manejar señales eeg, gráficas y procesamiento de frecuencia
from RawSignal import RawSignal
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.channels import DigMontage 
from mne.time_frequency import tfr_array_morlet

# clase que representa una señal eeg con procesamiento adicional encima de RawSignal
class EEGSignal:
    def __init__(self, signal: RawSignal, referencia, montaje=None):
        # chequeo que la señal sea del tipo correcto
        if not isinstance(signal, RawSignal):
            raise TypeError("La señal debe de ser una instancia de RawSignal")
        else:
            self.signal = signal

        # convierto a objeto Raw de MNE para poder usar sus herramientas
        self._raw_mne = mne.io.RawArray( self.signal.data.copy(), mne.create_info(self.signal.info.ch_names, self.signal.info.sfreq, ch_types='eeg'))
        self.sfreq = self.signal.sfreq

        # si hay montaje, lo cargo (puede ser nombre de archivo o DigMontage (objeto de mne))
        if montaje is not None:
            if isinstance(montaje, DigMontage):
                self.montage = montaje
            else:
                self.montage = mne.channels.read_custom_montage(montaje)
            self._raw_mne.set_montage(self.montage)
        else:
            self.montage = None

        # valido tipo de referencia (promedio, canal o laplaciano)
        if referencia.lower() not in ["canal", "promedio", "laplaciano"]:
            raise ValueError("La referencia debe ser: 'canal', 'promedio' o 'laplaciano'")
        else:
            self.referencia = referencia.lower()
        self.canal = None

    # permite cambiar la referencia y devuelve un nuevo EEGSignal
    def change_ref(self, newref, canal=None):
        """
        cambia la referencia de la señal
        """
        newref = newref.lower()
        if newref not in ["canal", "promedio", "laplaciano"]:
            raise ValueError("La nueva referencia debe ser: 'canal', 'promedio' o 'laplaciano'")

        raw_copy = self._raw_mne.copy()

        # referencia promedio
        if newref == "promedio":
            raw_ref, _ = mne.set_eeg_reference(raw_copy, ref_channels="average", copy=True)
            nueva_data = raw_ref.get_data()
            referencia = "promedio"
            canal_ref = None

        # referencia a canal específico
        elif newref == "canal":
            if canal is None:
                raise ValueError("Debe especificar un canal para la referencia tipo 'canal'.")
            if canal not in self.signal.info.ch_names:
                raise ValueError(f"El canal '{canal}' no existe en la señal.")
            else:
                raw_ref, _ = mne.set_eeg_reference(raw_copy, ref_channels=[canal], copy=True)
                nueva_data = raw_ref.get_data()
                referencia = "canal"
                canal_ref = canal

        # referencia laplaciana
        else:
            raw_ref = mne.preprocessing.compute_current_source_density(raw_copy, copy=True)
            nueva_data = raw_ref.get_data()
            referencia = "laplaciano"
            canal_ref = None

        # creo un nuevo RawSignal con los datos modificados
        nuevo_rawsignal = RawSignal(
            data=nueva_data,
            sfreq=self.signal.sfreq,
            info=self.signal.info,
            anotaciones=self.signal.anotaciones
        )

        # devuelvo nueva instancia con la nueva referencia
        nueva_instancia = EEGSignal(
            signal=nuevo_rawsignal,
            referencia=referencia,
            montaje=self.montage
        )
        nueva_instancia.canal = canal_ref

        print(f"Referencia cambiada a '{referencia}'.")
        return nueva_instancia

    # aplica filtro laplaciano clásico o CSD de MNE según si tengo o no el montaje
    def aplicar_filtro_laplaciano(self, vecinos_dict=None): 
        """
        aplica laplaciano espacial (usando CSD o vecinos)
        """
        if self._raw_mne.get_montage() is not None:
            csd_raw = mne.preprocessing.compute_current_source_density(self._raw_mne, copy = True)
            datos_filtrados = csd_raw.get_data()
        else:
            if vecinos_dict is None:
                raise ValueError("Se requiere vecinos_dict si no hay montaje.")

            datos = self.signal.data.copy()
            canales = self.signal.info.ch_names
            datos_filtrados = datos.copy()

            for i, canal in enumerate(canales):
                if canal in vecinos_dict:
                    vecinos = vecinos_dict[canal]
                    indices_vecinos = [canales.index(v) for v in vecinos if v in canales]
                    if indices_vecinos:
                        promedio_vecinos = np.mean(datos[indices_vecinos, :], axis=0)
                        datos_filtrados[i, :] = datos[i, :] - promedio_vecinos

        nueva_senal = RawSignal(data= datos_filtrados, sfreq=self.signal.sfreq, info=self.signal.info, anotaciones=self.signal.anotaciones)
        return EEGSignal(signal=nueva_senal, referencia="laplaciano", montaje= self.montage)
        
    # calcula el espectro de fourier de canales seleccionados, con opciones de log, promedio, suavizado y gráfico
    def calcular_espectro(self, canales=None, limite_frec=80, log=False, mean=False, plot=True, guardar_como=None, suavizar_ventana=5):
        """
        calcula el espectro de fourier para uno o varios canales
        """
        Fs = self.signal.info.sfreq
        todos_los_nombres = self.signal.info.ch_names
        n_canales, N = self.signal.data.shape

        # selecciono canales
        if canales is None:
            indices = range(n_canales)
            nombre_titulo = "Todos los canales"
        elif isinstance(canales, str):
            if canales not in todos_los_nombres:
                raise ValueError(f"Canal '{canales}' no encontrado.")
            indices = [todos_los_nombres.index(canales)]
            nombre_titulo = f"Canal {canales}"
        elif isinstance(canales, list):
            indices = []
            for c in canales:
                if c not in todos_los_nombres:
                    raise ValueError(f"Canal '{c}' no encontrado.")
                indices.append(todos_los_nombres.index(c))
            nombre_titulo = f"Canales {', '.join(canales)}"
        else:
            raise TypeError("El parámetro 'canales' debe ser None, str o list de str.")

        # fft
        freqs = np.fft.rfftfreq(N, 1 / Fs)
        espectros = []

        for i in indices:
            datos = self.signal.data[i, :]
            trf = np.fft.rfft(datos)
            espectro = np.abs(trf)
            espectros.append(espectro)

        espectros = np.array(espectros)

        if log:
            espectros = 20 * np.log10(espectros + 1e-12)
            ylabel = "Amplitud (dB)"
        else:
            ylabel = "Amplitud"

        espectro_mean = np.mean(espectros, axis=0)
        espectro_std = np.std(espectros, axis=0) if mean else None

        # recorto hasta limite de frecuencia
        if limite_frec is not None:
            mask = freqs <= limite_frec
            freqs = freqs[mask]
            espectros = espectros[:, mask]
            espectro_mean = espectro_mean[mask]
            if mean:
                espectro_std = espectro_std[mask]

        # suavizado por ventana móvil
        def suavizar(y, ventana=5):
            if ventana <= 1:
                return y
            return np.convolve(y, np.ones(ventana) / ventana, mode='same')

        espectro_mean = suavizar(espectro_mean, ventana=suavizar_ventana)
        if mean:
            espectro_std = suavizar(espectro_std, ventana=suavizar_ventana)

        # defino si tiene sentido mostrar el promedio
        mostrar_promedio = mean or len(indices) > 1

        # gráfico opcional
        if plot:
            plt.figure(figsize=(12, 5))
            paleta = ['#4C72B0', '#55A868', '#C44E52', '#8172B2','#CCB974', '#64B5CD', '#DD8452', '#937860']

            for i, esp in enumerate(espectros):
                color = paleta[i % len(paleta)]
                plt.plot(freqs, esp, color=color, linewidth=1.0, alpha=0.4)

            if mostrar_promedio:
                plt.plot(freqs, espectro_mean, color='black', linewidth=2, label='Promedio')

                if mean:
                    plt.fill_between(freqs, espectro_mean - espectro_std, espectro_mean + espectro_std,
                                    color='gray', alpha=0.25, label='±1 STD')
            plt.xlabel("Frecuencia (Hz)", fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            plt.title(f"Espectros de Fourier - {nombre_titulo}", fontsize=14, weight='bold')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.xlim(0, limite_frec)

            if mostrar_promedio:
                plt.legend(frameon=False, fontsize=10, loc='upper right')

            if log:
                ymin = np.percentile(espectro_mean, 5)
                ymax = np.percentile(espectro_mean, 99.5)
                plt.ylim(ymin, ymax)

            plt.tight_layout()

            if guardar_como:
                plt.savefig(guardar_como, dpi=300)
            plt.show()

        return freqs, espectro_mean, espectro_std

    # calcula la transformada tiempo-frecuencia usando ondas de morlet
    def calcular_tfr(self, canal, frecs=np.arange(2, 40, 2), n_cycles=5, use_log=True, plot=True):
        """
        calcula representación tiempo-frecuencia (morlet) para un canal
        """
        idx = self.signal.info.ch_names.index(canal)
        data = self._raw_mne.get_data(picks=[idx])[np.newaxis, :, :]  # shape (1, n_channels=1, n_times)

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

    # calcula la transformada de hilbert para canales seleccionados, con opciones de recorte, envolvente, fase, etc
    def calcular_hilbert(self, canales=None, envelope=False, return_magnitude_and_phase=False,
                        recorte_segundos=(0, 0), clip_percentiles=None, plot=True):
        """
        calcula la transformada de hilbert y permite devolver magnitud, fase o señal compleja
        """
        raw_copy = self._raw_mne.copy()

        # manejo de selección de canales
        picks = None
        if isinstance(canales, str):
            picks = [canales]
        elif isinstance(canales, list):
            picks = canales
        elif canales is not None:
            raise TypeError("El argumento 'canales' debe ser str, list[str] o None.")

        # aplico hilbert
        raw_hilbert = raw_copy.apply_hilbert(picks=picks, envelope=envelope)
        hilbert_data = raw_hilbert.get_data(picks=picks)

        # según opciones, defino salida
        if envelope:
            resultado = hilbert_data
        elif return_magnitude_and_phase:
            magnitud = np.abs(hilbert_data)
            fase = np.angle(hilbert_data)
            resultado = (magnitud, fase)
        else:
            resultado = hilbert_data

        # recorte en el tiempo
        t = raw_copy.times
        srate = raw_copy.info['sfreq']
        i_ini = int(recorte_segundos[0] * srate)
        i_fin = int(t.shape[0] - recorte_segundos[1] * srate)

        t = t[i_ini:i_fin]
        if isinstance(resultado, tuple):
            resultado = tuple(r[:, i_ini:i_fin] for r in resultado)
        else:
            resultado = resultado[:, i_ini:i_fin]

        # gráfico si hay un solo canal
        if plot and resultado.shape[0] == 1:
            y = resultado[0] if not isinstance(resultado, tuple) else resultado[0][0]

            if clip_percentiles is not None:
                ymin = np.percentile(y, clip_percentiles[0])
                ymax = np.percentile(y, clip_percentiles[1])
            else:
                ymin, ymax = np.min(y), np.max(y)

            plt.figure(figsize=(10, 4))
            plt.plot(t, y, color='mediumvioletred')
            plt.title(f"Transformada de Hilbert - Canal {picks[0]}")
            plt.xlabel("Tiempo (s)")
            plt.ylabel("Amplitud" if envelope else "Real (Hilbert)")
            plt.ylim(ymin, ymax)
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.show()

        return resultado