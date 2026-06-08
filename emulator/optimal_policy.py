"""
OptimalOverlapPolicy — теоретически лучшая стратегия для задачи 1.

Условия задачи 1:
  - номинальные a1, c1, a2, c2 известны
  - реальное время = nominal * U(0.8, 1.2)
  - два infer не параллельны, prepare — можно

Ключевые факты:
  1. throughput(b) = b / T_infer(b) монотонно растёт → всегда берём b_max
  2. prepare можно перекрывать с infer предыдущего батча
  3. идеальный старт prepare: ровно за T_prepare до конца infer

Два ограничения на момент закрытия батча:
  A) overlap:  t_close = infer_free_at - T_prepare_nominal(b)
               → prepare заканчивается одновременно с infer, idle = 0
  B) SLA:      t_close ≤ oldest_arrival + SLA - worst*(T_prepare + T_infer)
               → самый старый запрос успеет уложиться в SLA

  Берём min(A, B) — это дедлайн. Если он уже прошёл — закрываем сейчас.

Параметр safety (>= 1.0):
  Умножает T_prepare в формуле overlap, запуская prepare чуть раньше.
  safety=1.0 → запуск по номиналу, в среднем 50% попаданий без idle
  safety=1.2 → запуск на 20% раньше, почти всегда prepare успевает,
               но чуть больше idle времени перед infer
"""

from typing import Optional
from models import SystemState, Decision
from policies import BasePolicy


class OptimalOverlapPolicy(BasePolicy):
    """
    Стратегия:
      - размер батча всегда b_max (максимизирует throughput)
      - момент закрытия = min(overlap_deadline, sla_deadline)
      - если система idle — закрываем сразу (нет смысла ждать)
    """

    def __init__(self, safety: float = 1.0, max_batch_size: Optional[int] = None):
        """
        safety: множитель для T_prepare в расчёте overlap-дедлайна.
                1.0 = номинал, 1.2 = с запасом 20% на variance.
        """
        self.safety = safety
        self.max_batch_size = max_batch_size

    def name(self) -> str:
        return f"Optimal(safety={self.safety})"

    def decide(self, state: SystemState) -> Decision:
        if not state.queue:
            return Decision(close_batch_at=None, batch_size=None)

        p = state.params
        worst = self.safety * (1.0 + p.variance)

        # Всегда берём b_max — throughput монотонно растёт с размером батча
        cap = self._cap(state, self.max_batch_size)
        b = min(len(state.queue), cap)

        # --- Ограничение A: overlap с текущим infer ---
        # Когда освободится infer с учётом уже запущенных батчей
        if state.infer_busy:
            infer_free = state.infer_end_time + state.committed_infer_nominal
        else:
            infer_free = state.now + state.committed_infer_nominal

        # Хотим закончить prepare ровно когда infer освободится
        # → закрываем батч за safety*T_prepare до этого момента
        overlap_close = infer_free - self.safety * p.t_prepare_nominal(b)

        # --- Ограничение B: SLA для самого старого запроса ---
        oldest_arrival = state.queue[0].arrival_time
        # Самый пессимистичный суммарный процессинг
        worst_processing = worst * (p.t_prepare_nominal(b) + p.t_infer_nominal(b))
        sla_close = oldest_arrival + state.sla_ms - worst_processing

        # Берём более ранний из двух дедлайнов
        close_at = min(overlap_close, sla_close)

        # Система простаивает — нет смысла ждать overlap-дедлайна,
        # но SLA-дедлайн всё равно соблюдаем
        system_idle = not state.infer_busy and state.committed_count == 0
        if system_idle:
            close_at = sla_close

        # Если дедлайн уже прошёл или наступил — закрываем сейчас
        if close_at <= state.now:
            return Decision(close_batch_at=state.now, batch_size=b)

        return Decision(close_batch_at=close_at, batch_size=None)