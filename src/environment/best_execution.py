"""Entorno de ejecución para los episodios de best execution."""
from typing import Tuple

import numpy as np
import pandas as pd
from copy import deepcopy

from .config import EnvironmentConfig
from src.rewards import RewardContext, RewardStrategy


class BestExecutionEnv:

    def __init__(self, data, config: EnvironmentConfig, reward_strategy: RewardStrategy):
        """Inicialización de la clase del entorno que simula
        el libro de ordenes.
        ----------------------------------------------------
        Input:
            - data: 
                Dataframe con los datos previamente
                agrupados del libro de órdenes.
            
            - config:
                Configuración tipada de la generación de episodios.

            - reward_strategy:
                Estrategia de recompensa que se utilizará para calcular
                la recompensa de cada acción tomada por el agente.
        ----------------------------------------------------
        Variables Internas:
            - episode_bins:
                Número de bines (steps) del episodio.
            - episode_full_len:
                Es igual a look_back + episode_bins.
            - vol_care:
                Volumen total (en títulos) de la orden care.
            - actions_fn:
                Diccionario con las posibles acciones del agente.
                Las claves acceden a la función que evalúa la acción
                tomada por el agente.
            - n_actions:
                Número de acciones posibles.
            - n_features:
                Número de características de los estados.
            - episode:
                Dataframe que contiene los steps y estados del episodio.
            - episode_full:
                Es el episode añadiendo el look_back antes del comienzo 
                del episodio.
            - episode_vwap:
                VWAP de mercado al final del episodio.
            - market_ep_vol:
                Volumen (títulos) ejecutado por el mercado en cada bin del episodio.
            - state_pos:
                Número de step en el que nos encontramos.
            - exec_vol:
                Acumulado de títulos ejecutados por el algoritmo.
            - actions_hist:
                Lista de acciones tomadas por el algoritmo en cada step.
            - market_vwap_hist:
                Lista de VWAP de mercado en cada step.
            - reward_hist:
                Lista de rewards obtenidas en cada step.
            - price_hist:
                Lista de precios ejecutados en cada step.
            - vol_hist:
                Lista de títulos ejecutados en cada step.                
        """

        if not isinstance(config, EnvironmentConfig):
            raise TypeError("config must be an EnvironmentConfig instance")

        if not isinstance(reward_strategy, RewardStrategy):
            raise TypeError("reward_strategy must be a RewardStrategy instance")

        # Fixed params
        self.data = data
        self.config = config
        self.reward_strategy = reward_strategy
        self.look_back = config.look_back
        self._rng = np.random.default_rng(config.seed)
        self.episode_bins = None
        self.episode_full_len = None
        self.vol_care = None
        
        self.actions_fn = {
            0: self._do_nothing,
            1: self._agg_action,
        }
        
        self.n_actions = len(self.actions_fn)
        self.n_features = self._detect_num_feat()

        # Data variables
        self.episode = None
        self.episode_full = None
        
        # Env variables
        self.episode_vwap = None
        self.market_ep_vol = None
        self.state_pos = 0
        self.exec_vol = 0
        self.actions_hist = []
        self.algo_vwap_hist = []
        self.market_vwap_hist = []
        self.reward_hist = []
        self.price_hist = []
        self.vol_hist = []
        
    def _detect_num_feat(self) -> int:
        """Detecta las dimensiones sin consumir el estado aleatorio."""
        rng_state = deepcopy(self._rng.bit_generator.state)

        try:
            observation = self.reset()
            return int(observation.shape[0])
        finally:
            self._rng.bit_generator.state = rng_state
            self._reset_env_episode_params()
        
    def _reset_env_episode_params(self):
        """
        Reset del episodio e inicialización de los parámetros.
        Las variables internas vuelven a sus valores originales.
        """
        self.episode_full_len = None
        self.episode = None
        self.episode_full = None
        self.episode_vwap = None
        self.market_ep_vol = None
        self.state_pos = 0
        self.algo_vwap = 0
        self.exec_vol = 0
        self.actions_hist = []
        self.algo_vwap_hist = []
        self.market_vwap_hist = []
        self.reward_hist = []
        self.price_hist = []
        self.vol_hist = []
        self.episode_bins = None
        self.vol_care = None
        
    def _generate_episode_params(self) -> None:
        """(1) Se determinan las características de la orden a ejecutar.
        La orden queda definida por:
         - episode_bins:
             Número entero aleatorio dentro del rango configurado.
         - vol_care:
             Fracción aleatoria configurada multiplicada por episode_bins y
             convertida a un entero.
        """
        self.episode_bins = int(
            self._rng.integers(
                low=self.config.min_episode_bins,
                high=self.config.max_episode_bins + 1,
            )
        )
        care_fraction = float(
            self._rng.uniform(
                low=self.config.min_care_fraction,
                high=self.config.max_care_fraction,
            )
        )
        self.vol_care = int(care_fraction * self.episode_bins)
        self.episode_full_len = self.episode_bins + self.look_back
        
    def _generate_episode(self) -> None:
        """(2) Selecciona un día y una posición inicial válidos."""
        self._generate_episode_params()

        if not hasattr(self.data, "items"):
            raise TypeError("data must be a mapping of days to DataFrames")

        eligible_days = []
        for day, data_day in self.data.items():
            if not isinstance(data_day, pd.DataFrame):
                raise TypeError(f"data for day {day!r} must be a pandas DataFrame")
            if len(data_day) >= self.episode_full_len:
                eligible_days.append(day)

        if not eligible_days:
            raise ValueError(
                "no day contains enough rows for look_back "
                f"({self.look_back}) and episode_bins ({self.episode_bins})"
            )

        selected_day = eligible_days[
            int(self._rng.integers(low=0, high=len(eligible_days)))
        ]
        data_day = self.data[selected_day]

        min_hour_pos = self.look_back
        max_hour_pos = len(data_day) - self.episode_bins
        hour_pos = int(
            self._rng.integers(low=min_hour_pos, high=max_hour_pos + 1)
        )

        initial_position = hour_pos - self.look_back
        final_position = hour_pos + self.episode_bins

        self.episode_full = data_day.iloc[
            initial_position:final_position
        ].copy()
        self.episode = data_day.iloc[hour_pos:final_position].copy()

    def reset(self) -> np.array:
        """Reinicialización del episodio junto con los parámetros.
        Devuelve la primera observación del nuevo episodio.
        """  
        self._reset_env_episode_params()     
        self._generate_episode()
        self._compute_episode_market_feat()
        
        return self.observation_builder()
    
    def observation_builder(self) -> np.ndarray:
        """(3) Función para la construcción de las observaciones del estado.
            ------------------------------------------------------------
            Default:
                - Primera característica es la fracción de tiempo restante.
                - Segunda característica es la fracción de volumen restante.
        """
        time_left = (self.episode_bins - self.state_pos) / self.episode_bins
        vol_left = (self.vol_care - self.exec_vol) / self.vol_care

        return np.array([time_left, vol_left], dtype=np.float32)
    
    def _compute_episode_market_feat(self) -> Tuple[float, pd.Series]:
        """(4) Cálculo de los valores VWAP y Market Vol del episodio.
        Como no tenemos las ejecuciones de mercado, asumimos que el 
        precio es el mid price de cada step.
        """
        # Calcula el mid price utilizando ask1 y bid1 de episode
        # Opcional: Utiliza un precio más realista para el mkt VWAP
        mid = (self.episode["ask1"] + self.episode["bid1"]) / 2
        # Calcula market_ep_vol
        self.market_ep_vol = self.episode["cumvol"].diff().fillna(0.0)
        if (self.market_ep_vol < 0).any():
            raise ValueError("cumvol must be non-decreasing during the episode")
        # Calcula el volumen acumulado del mercado en todo el episodio
        cum_vol = self.market_ep_vol.sum()
        # Validamos que el volumen acumulado del mercado es mayor que 0
        if cum_vol <= 0:
            raise ValueError(
                f"Market volume for the episode is {cum_vol}, "
                "which is not greater than 0"
            )

        # Calcula el episode_vwap
        # Clue: mid de cada bin por el volumen del bin siguiente (usa .values para que pandas no alinee por índice)
        prices = mid.iloc[:-1].to_numpy()
        volumes = self.market_ep_vol.iloc[1:].to_numpy()

        self.episode_vwap = np.sum(prices * volumes) / cum_vol
        
        return self.episode_vwap, self.market_ep_vol
    
    def _compute_algo_vwap(self) -> float:
        """Calcula el VWAP acumulado de las ejecuciones del algoritmo."""
        prices = np.asarray(self.price_hist, dtype=float)
        volumes = np.asarray(self.vol_hist, dtype=float)

        total_volume = volumes.sum()
        if total_volume == 0:
            return 0.0

        return float(np.sum(prices * volumes) / total_volume)

    def _compute_reward_context(self, state_pos: int) -> RewardContext:
        """Construye un RewardContext con el estado actual del episodio."""
        return RewardContext(
            execution_price=self.price_hist[-1],
            execution_volume=self.vol_hist[-1],
            episode_vwap=self.episode_vwap,
            state_pos=state_pos,
            episode_bins=self.episode_bins,
            executed_volume=self.exec_vol,
            care_volume=self.vol_care,
        )
    
    def _compute_reward(self) -> float:
        """Calcula la recompensa de la última acción registrada.

        Construye un ``RewardContext`` con el precio y volumen de la última
        acción, junto con el estado acumulado del episodio después de
        aplicarla. El cálculo se delega en la ``RewardStrategy`` configurada,
        por lo que el entorno no contiene lógica específica de recompensa.
        """
        reward_context = self._compute_reward_context(self.state_pos)
        reward = self.reward_strategy.compute(reward_context)
        return reward
    
    def _compute_stop_conditions(self) -> Tuple[bool, bool]:
        """(10) Define las condiciones de parada del episodio
        Return:
            Tiempo agotado, orden completada
        """
        # Calcula las variables de parada y devuélvelas en el orden apropiado
        is_bins_complete = self.state_pos >= self.episode_bins
        is_ord_complete = self.exec_vol >= self.vol_care
        return is_bins_complete, is_ord_complete
    
    def _compute_done_reward(self) -> float:
        """(12)
        """
        # Recompensa terminal, después del incremento de state_pos efectuado por step()
        reward_context = self._compute_reward_context(self.state_pos - 1)
        done_reward = self.reward_strategy.compute_terminal(reward_context)
        return done_reward

    def _record_execution(self, price: float, volume: int) -> float:
        """Registra una acción y devuelve su recompensa."""
        self.price_hist.append(price)
        self.vol_hist.append(volume)
        self.exec_vol += volume

        algo_vwap = self._compute_algo_vwap()
        self.algo_vwap_hist.append(algo_vwap)

        return self._compute_reward()
    
    def _agg_action(self) -> float:
        """(7) Acción agresiva de compra de un título a precio de episode['ask1'].
        Devolvemos el reward asociado a esa acción.
        """
        price = float(self.episode["ask1"].iloc[self.state_pos])
        return self._record_execution(price, volume=1)

    def _do_nothing(self) -> float:
        """(9) No hacer nada y devolvemos el reward asociado a la acción
        """
        return self._record_execution(price=0.0, volume=0)
    
    def _compute_market_vwap(self) -> float:
        """(5) Cálculo del VWAP del mercado hasta el step actual.
        """
        # Establece un precio para el vol ejecutado por el mkt en cada step
        mid_p = (self.episode["ask1"] + self.episode["bid1"]) / 2
        #Para el primer estep, el vwap del mkt es el mid price del primer step
        if self.state_pos == 0:
            return float(mid_p.iloc[0])
        # Calcula todos los vwap del mkt hasta el step actual incluido
        v = self.market_ep_vol.iloc[1:self.state_pos + 1]
        p_arr = mid_p.iloc[:self.state_pos].to_numpy()
        v_arr = v.to_numpy()
        sum_vol = v_arr.sum()
        # Si el mkt vol hasta el step == 0, devuelve el último precio hasta el step
        if sum_vol == 0:
            return float(mid_p.iloc[self.state_pos])
        # Calcula y devuelve el vwap acumulado hasta el step
        market_vwap = float(np.sum(p_arr * v_arr) / sum_vol)
        return market_vwap
    
    def _compute_done(self) -> bool:
        """Indica si el episodio ha terminado."""
        return any(self._compute_stop_conditions())

    def step(self, action) -> Tuple[np.array, float, bool, dict]:
        """ Evalúa la acción, calcula la recompensa, devuelve el 
        nuevo estado y si el episodio ha terminado.
        """
        market_vwap = self._compute_market_vwap()
        
        act_fn = self.actions_fn.get(action)
        if act_fn is None:
            raise ValueError(
                f"Invalid action {action}. Valid actions {self.actions_fn.keys()}"
            )

        reward = act_fn()
        self.actions_hist.append(action)
        self.market_vwap_hist.append(market_vwap)

        self.state_pos += 1
        done = self._compute_done()

        if done:
            reward += self._compute_done_reward()

        self.reward_hist.append(reward)

        if done:
            return None, reward, done, {}

        observation = self.observation_builder()
        return observation, reward, done, {}
    
    def action_sample(self) -> int:
        """Devuelve una acción aleatoria usando la distribución del episodio."""
        execution_probability = self.vol_care / self.episode_bins

        return int(
            self._rng.choice(
                [0, 1],
                p=[1 - execution_probability, execution_probability],
            )
        )

    def stats_df(self):
        """Información para el gráfico de resultados de la ejecución
        """
        
        my_df = pd.DataFrame(
            {"vwap": self.algo_vwap_hist, "vol": self.vol_hist},
            index=list(self.episode.index)[:len(self.algo_vwap_hist)]
        )
        my_df = my_df.reindex(self.episode.index)
        my_df["vol"] = my_df["vol"].fillna(0)
        my_df["vwap"] = my_df["vwap"].ffill()
            
        
        p = self.episode["ask1"]
        v = self.episode["cumvol"].diff().shift(-1)
        last_v = self.episode_full["cumvol"].diff().iloc[-1]
        v.iloc[-1] = last_v
        market_vwap = (p * v).cumsum() / v.cumsum()
        market_df = pd.DataFrame(
            {"vwap": market_vwap, "vol": v},
            index=v.index
        )
        
        mpx = (self.episode["ask1"] + self.episode["bid1"]) / 2
        
        return my_df, market_df, mpx

