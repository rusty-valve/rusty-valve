"""Lossless compressed scoring observations and versioned offline assessment.

Writes the already-solved environment callback at every physics step. It does
not call a simulator, reconstruct contact forces, or subsample observations.
The footer is required: an interrupted/truncated file cannot establish a pass.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path

from .scoring_v25 import OnlineScorer, ScoreConfig
from .scoring_v24 import OnlineScorer as OnlineScorerV24
from .scoring_v23 import OnlineScorer as OnlineScorerV23

FILENAME = 'physical_step_evidence.jsonl.gz'
FORMAT = 'valve_solved_step_evidence_v1'


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    return value


class EvidenceWriter:
    def __init__(self, path, *, variant, station, physics_dt_s, procedure_version,
                 face_constraint_applicable, face_evidence_complete, config=None, source=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._raw = self.path.open('xb')
        self._stream = gzip.GzipFile(fileobj=self._raw, mode='wb', compresslevel=3, mtime=0)
        self.count = 0; self.first_time = self.last_time = None
        self.continuous = True; self.closed = False; self.summary = None
        self.dt = float(physics_dt_s)
        self._write(dict(type='header', format=FORMAT, variant=variant, actual_station=station,
            physics_dt_s=self.dt, observation_dt_s=self.dt, procedure_version=procedure_version,
            data_kind='recorded_simulation_every_solved_physics_step',
            face_constraint_applicable=bool(face_constraint_applicable),
            face_evidence_complete=bool(face_evidence_complete), contact_evidence_complete=True,
            source=source, score_config=asdict(config or ScoreConfig())))

    def _write(self, row):
        self._stream.write((json.dumps(_plain(row), ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n').encode('utf-8'))

    def observe(self, time_s, opening_deg, speed_deg_s, contacts, metadata=None):
        if self.closed:
            raise RuntimeError('Cannot append evidence after the footer')
        t = float(time_s)
        if self.first_time is None:
            self.first_time = t
            self.continuous = abs(t) <= 1e-9
        elif not (0 < t-self.last_time <= self.dt+max(1e-7, self.dt*.01)):
            self.continuous = False
        self.last_time = t; self.count += 1
        self._write(dict(type='sample', time_s=t, opening_deg=float(opening_deg),
            speed_deg_s=float(speed_deg_s), contacts=contacts, metadata=metadata or {}))

    def finish(self, *, terminal_time_s, execution_complete, score=None):
        if self.closed:
            return self.summary
        complete = bool(execution_complete and self.continuous and self.count >= 2
                        and abs(float(terminal_time_s)-self.last_time) <= 1e-7)
        self.summary = dict(filename=self.path.name, format=FORMAT,
            samples=self.count, first_time_s=self.first_time, last_time_s=self.last_time,
            physics_dt_s=self.dt, record_complete=complete,
            execution_complete=bool(execution_complete), continuous=self.continuous,
            full_contact_vectors_saved=True, physical_steps_reconstructed=0)
        self._write(dict(type='footer', **self.summary, score=score))
        self._stream.close(); self._raw.close(); self.closed = True
        return self.summary


def _read_log(path, *, rescore, procedure_version='A1-v2.5'):
    scorers = {'A1-v2.3': OnlineScorerV23, 'A1-v2.4': OnlineScorerV24, 'A1-v2.5': OnlineScorer}
    if rescore and procedure_version not in scorers:
        raise ValueError('Offline evidence scoring supports A1-v2.3/v2.4/v2.5')
    header = footer = scorer = None; count = 0; first = last = None; continuity = True
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if footer is not None:
                raise ValueError('Data exists after the evidence footer')
            kind = row.get('type')
            if header is None:
                if kind != 'header' or row.get('format') != FORMAT:
                    raise ValueError('Missing/unsupported evidence header')
                header = row
                if rescore:
                    scorer = scorers[procedure_version](row['variant'], actual_station=row['actual_station'],
                        physics_dt_s=row['physics_dt_s'], observation_dt_s=row['observation_dt_s'],
                        face_constraint_applicable=row['face_constraint_applicable'],
                        face_evidence_complete=row['face_evidence_complete'],
                        contact_evidence=row['contact_evidence_complete'],
                        config=ScoreConfig(**row['score_config']), source=str(path))
            elif kind == 'sample':
                t = float(row['time_s']); dt = header['physics_dt_s']
                if first is None:
                    first = t; continuity = abs(t) <= 1e-9
                elif not (0 < t-last <= dt+max(1e-7, dt*.01)):
                    continuity = False
                last = t; count += 1
                if scorer is not None:
                    scorer.observe(t, row['opening_deg'], row['speed_deg_s'], row['contacts'], row.get('metadata', {}))
            elif kind == 'footer':
                footer = row
            else:
                raise ValueError('Unexpected evidence record')
    if header is None or footer is None:
        raise ValueError('Evidence lacks a complete header/footer; no score was issued')
    if footer['samples'] != count or footer['first_time_s'] != first or footer['last_time_s'] != last:
        raise ValueError('Evidence footer disagrees with actual sample inventory')
    complete = bool(footer['record_complete'] and footer['execution_complete'] and continuity and count >= 2)
    summary = dict(format=FORMAT, filename=Path(path).name, samples=count,
                   first_time_s=first, last_time_s=last, record_complete=complete,
                   gzip_read_to_eof=True, full_contact_vectors_saved=True,
                   physical_steps_reconstructed=0, original_procedure_version=header['procedure_version'])
    if scorer is not None:
        if not complete:
            scorer.coverage_ok = False
            scorer.waits.coverage_ok = False
        result = scorer.result()
        result.update(physical_step_evidence=dict(summary), data_kind=header['data_kind'],
                      offline_rescore_only=True, original_procedure_version=header['procedure_version'])
        summary['score'] = result
    return summary


def inspect_evidence_log(path):
    """Read gzip to EOF and validate every saved step without running physics."""
    return _read_log(path, rescore=False)


def rescore_evidence_log(path, *, procedure_version='A1-v2.5'):
    return _read_log(path, rescore=True, procedure_version=procedure_version)['score']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--procedure-version', choices=('A1-v2.3', 'A1-v2.4', 'A1-v2.5'),
                        default='A1-v2.5', help='Explicit offline scoring policy; source evidence is never modified')
    args = parser.parse_args()
    score = rescore_evidence_log(args.path, procedure_version=args.procedure_version)
    payload = json.dumps(score, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(payload)
    print(payload, end='')


if __name__ == '__main__':
    main()
