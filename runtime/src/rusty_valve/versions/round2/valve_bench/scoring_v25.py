"""v2.5: a physically identified rim on the working side is valid contact.

Inherits v2.4's target, hold, collision, terminal-contact and coverage rules.
Only finite working-side hole-rim geometry gets the revised face attribution;
unidentified geometry and positively identified nonworking faces stay strict.
Historical v2.4/v2.3 scorers retain their own default face policy.
"""
from __future__ import annotations

from .scoring_v24 import OnlineScorer as V24Scorer, ScoreConfig, VARIANTS, _score_record
from .tool_face_evidence import assess_contact_drive


class OnlineScorer(V24Scorer):
    def _assess_contact_drive(self, evidence):
        return assess_contact_drive(evidence, allow_working_hole_rims=True)

    def result(self, *, final_target_override=None):
        result = super().result(final_target_override=final_target_override)
        result.update(schema_version='2.5', procedure_version='A1-v2.5',
                      assessment_scope='所有工位统一：实体工作侧孔口边缘可承力，其余目标、保持、碰撞及接触记录要求沿用v2.4')
        result['rules']['working_face']['geometry_policy'] = (
            'working-side physical hole rims accepted; unknown geometry and nonworking faces unchanged')
        result['scoring_notes'].append('v2.5认可实体工作侧的孔口边缘接触；不把任意未知接触改成有效工作面。其它评分要求沿用v2.4。')
        return result


def score_record(record, config=None):
    return _score_record(record, config, scorer_class=OnlineScorer)
