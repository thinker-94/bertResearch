# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD 1.1 and SQuAD 2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import random
import modeling
import optimization
import tokenization
import six
import tensorflow as tf

print("dd")
#https://daeson.tistory.com/256 tensorflow flags 사용법 & 설명
flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", 'bert_config.json',
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", 'vocab.txt',
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", '/bert',
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string("train_file", "KorQuAD_v1.0_train.json",
                    "SQuAD json for training. E.g., train-v1.1.json")

flags.DEFINE_string(
    "predict_file", None,
    "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 1, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 1,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 1.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")

flags.DEFINE_bool(
    "version_2_with_negative", False,
    "If true, the SQuAD examples contain some that do not have an answer.")

flags.DEFINE_float(
    "null_score_diff_threshold", 0.0,
    "If null_score - best_non_null is greater than the threshold predict null.")

class SquadExample(object):

  """A single training/test example for simple sequence classification.

     For examples without an answer, the start and end position are -1.
  """

  def __init__(self,
               qas_id,
               question_text,
               doc_tokens,
               orig_answer_text=None,
               start_position=None,
               end_position=None,
               is_impossible=False):
    self.qas_id = qas_id
    self.question_text = question_text
    self.doc_tokens = doc_tokens
    self.orig_answer_text = orig_answer_text
    self.start_position = start_position
    self.end_position = end_position
    self.is_impossible = is_impossible

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
    s += ", question_text: %s" % (
        tokenization.printable_text(self.question_text))
    s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
    if self.start_position:
      s += ", start_position: %d" % (self.start_position)
    if self.start_position:
      s += ", end_position: %d" % (self.end_position)
    if self.start_position:
      s += ", is_impossible: %r" % (self.is_impossible)
    return s


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               example_index,
               doc_span_index,
               tokens,
               token_to_orig_map,
               token_is_max_context,
               input_ids,
               input_mask,
               segment_ids,
               start_position=None,
               end_position=None,
               is_impossible=None):
    self.unique_id = unique_id
    self.example_index = example_index
    self.doc_span_index = doc_span_index
    self.tokens = tokens
    self.token_to_orig_map = token_to_orig_map
    self.token_is_max_context = token_is_max_context
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.start_position = start_position
    self.end_position = end_position
    self.is_impossible = is_impossible


# def read_squad_examples(input_file, is_training): 수정
def read_squad_examples(is_training):
  """Read a SQuAD json file into a list of SquadExample."""
  # with tf.gfile.Open(input_file, "r") as reader: 수정
  #   input_data = json.load(reader)["data"] 수정

  input_data = [{'paragraphs': [{'context': '원격 의료는 외딴 지역이나 원거리에 살고 있는 환자들에게 유익하다. 그들은 직접 의사나 '
                                            '전문의들을 만나러 멀리까지 갈 필요 없이 치료를 받을 수 있다. 최근의 모바일 협업 '
                                            '기술의 발전은 같은 지역의 경우, 다양한 장소에 있는 의료인들이 정보를 공유하고 '
                                            '환자들의 문제에 관해 의논하는 것을 가능하게 만들었다. 모바일 기술을 이용한 원격 환자 '
                                            '모니터링은 외래 환자들의 방문 필요성을 줄여주고, 원격 처방의 검증과 약물 투여 관리를 '
                                            '가능하게 하여 건강 관리의 전반적인 비용을 잠재적으로 상당히 줄일 수 있게 된다. 또, '
                                            '원격의료는 근무자들에게 전문가들을 보고 배울 수 있게 하고, 최적의 실습을 훨씬 쉽게 '
                                            '공유함으로써, 의학 교육 효과도 있다.',
                                 'qas': [{'answers': [{'answer_start': 0, 'text': '원격 의료'}],
                                          'id': '6526552-0-0',
                                          'question': '외딴 지역이나 원거리에 살고 있는 환자들에게 유익한 의료의 방법은?'},
                                         {'answers': [{'answer_start': 89,
                                                       'text': '모바일 협업 기술'}],
                                          'id': '6526552-0-1',
                                          'question': '많은 의료인들이 정보를 공유하고 환자들의 문제에 관해 의논 가능하게 '
                                                      '한 것은?'},
                                         {'answers': [{'answer_start': 89,
                                                       'text': '모바일 협업 기술'}],
                                          'id': '6542182-0-0',
                                          'question': '다양한 장소에서 의료인의 정보를 고유하고 환자의 문제를 의논할 수 '
                                                      '있도록 만든 기술은?'},
                                         {'answers': [{'answer_start': 184,
                                                       'text': '원격 환자 모니터링'}],
                                          'id': '6542182-0-1',
                                          'question': '외래 환자의 병원 방문을 줄여주고 원격처방의 검증과 약물 투여관리를 '
                                                      '가능하게 하는 것은?'},
                                         {'answers': [{'answer_start': 0, 'text': '원격 의료'}],
                                          'id': '5847018-0-0',
                                          'question': '외딴 지역이나 원거리에 살고 있는 환자들에게 필요한 의료 서비스는 '
                                                      '무엇인가?'},
                                         {'answers': [{'answer_start': 344, 'text': '의학 교육'}],
                                          'id': '5847018-0-1',
                                          'question': '원격의료가 전공자들에게 갖는 이점은?'}]},
                                {'context': '원격 의료의 단점은 원격 통신과 데이터 관리 장비의 비용, 그리고 원격 진료를 사용할 '
                                            '의료인들을 위한 기술 훈련의 비용에 있다. 가상 의료는 잠재적으로 의료 전문가들과 환자 '
                                            '간의 인간적인 상호 소통을 줄어들게 하고, 등록된 전문가의 부재 시에 의료 서비스를 '
                                            '제공할 경우의 위험성 증가, 또 전자 저장 매체로의 저장과 전송 과정에서 보호된 건강 '
                                            '정보를 위태롭게 할 위험을 수반하고 있다. 또한 가상의 상호 작용을 통하여 환자를 '
                                            '판단하여 치료하는 어려움 때문에 시간적인 효율성도 사실상 떨어진다. 예를 들어, 보통의 '
                                            '일반적인 피부과 상담은 15분이 걸린다면, 원격 피부과 상담은 30분이 걸린다. '
                                            '추가적으로, 관련된 임상 정보들을 접할 기회도 줄어들었고, 영상이나 환자 진행 경과 '
                                            '보고서 등 전송된 기록의 상태가 안 좋을 수 있어서 의사들에게 전달 해줄 환자 케어 '
                                            '정보는 품질을 보장하는 데 어려움이 있다. 원격 의료를 수행하는데 있어서 다른 '
                                            '장애물에는 몇몇 원격 의료 관행에 있어서 명확하지 않은 법적 규제와 또 어떤 분야에서는 '
                                            '정부 프로그램이나 보험 업자에게 손해 배상을 청구하기 어려운 점 등이 있다.',
                                 'qas': [{'answers': [{'answer_start': 420,
                                                       'text': '환자 케어 정보'}],
                                          'id': '6542182-1-0',
                                          'question': '원격 의료의 단점으로 전송된 기록의 상태가 좋지 않을 시, 문제가 '
                                                      '발생할 수 있는 부분은?'},
                                         {'answers': [{'answer_start': 64, 'text': '비용'}],
                                          'id': '6542182-1-1',
                                          'question': '원격 의료를 위한 원격 통신, 테이터 관리 장비, 의료인의 기술 '
                                                      '훈련에 있어 공통적으로 문제가 되는 것은?'},
                                         {'answers': [{'answer_start': 29, 'text': '비용'}],
                                          'id': '5847018-1-0',
                                          'question': '원격 의료의 단점은 무엇인가?'},
                                         {'answers': [{'answer_start': 256,
                                                       'text': '시간적인 효율성'}],
                                          'id': '5847018-1-1',
                                          'question': '원격 의료가 직접 의료보다 떨어지는 것은?'}]},
                                {'context': '축적 전송 방식은 의학 데이터(의학 영상, 생체 신호 등)를 수집하고, 오프라인에서도 '
                                            '종합 진단을 내려 의사나 전문의에게 편한 시간 때에 전송할 수 있다. 수신 측의 '
                                            '단말이나 발신 단말이 동시에 있지 않아도 통신이 가능하다. 피부과, 방사선학, 병리학 '
                                            '등은 동시에 통신하지 않는 원격 의료를 더 하기 쉽게 만드는 흔한 전문 분야이다. '
                                            '적절하게 구성된 진료 기록(가급적이면 전자식으로)도 전송되어야 한다. 전통적인 환자를 '
                                            '직접 대하는 방법과 원격 의료를 통해 환자를 대하는 것의 중요한 차이는 실질적인 몸의 '
                                            '진찰과 환자의 내력이 빠진 것이다. 축적 전송 방식은 임상 의사들이 환자의 내력과 몸을 '
                                            '진찰하지 못하는 대신에 음성/영상 정보를 필요로 한다.',
                                 'qas': [{'answers': [{'answer_start': 0, 'text': '축적 전송 방식'}],
                                          'id': '6526552-2-0',
                                          'question': '의학 데이터를 수집하고 오프라인에서도 의사나 전문의에게 전송할 수 '
                                                      '있는 방식은?'},
                                         {'answers': [{'answer_start': 345, 'text': '음성/영상 '}],
                                          'id': '6526552-2-1',
                                          'question': '임상 의사들이 환자의 내력과 몸을 진찰하지 못하는 대신에 무엇을 '
                                                      '필요로 할까?'},
                                         {'answers': [{'answer_start': 0, 'text': '축적 전송 방식'}],
                                          'id': '6542182-2-0',
                                          'question': '의학 데이터를 수집하고 오프라인에서도 종합진단을 내려 의료인에게 편한 '
                                                      '시간에 보낼 수 있는 전송 방식은?'},
                                         {'answers': [{'answer_start': 196, 'text': '진료 기록'}],
                                          'id': '6542182-2-1',
                                          'question': '축적 전송 방식을 통한 원격 진료에서 환자의 의학 데이터와 함께 '
                                                      '전송해야 하는 것은?'},
                                         {'answers': [{'answer_start': 0, 'text': '축적 전송 방식'}],
                                          'id': '5847018-2-0',
                                          'question': '수신 측의 단말이나 발신 단말이 동시에 있지 않아도 통신이 가능한 '
                                                      '전송 방식은?'},
                                         {'answers': [{'answer_start': 275,
                                                       'text': '실질적인 몸의 진찰과 환자의 내력이 빠진 것'}],
                                          'id': '5847018-2-1',
                                          'question': '환자를 대하는 것에 있어 직접 의료와 원격 의료의 중요한 차이는 '
                                                      '무엇인가?'}]},
                                {'context': '집에 있는 환자들의 모니터링은 혈압계와 같이 알려진 기기들을 사용하고 의료진에게 정보를 '
                                            '전송해 주는데, 빠르게 성장하는 최근에 생겨난 서비스들이다. 이러한 원격 모니터링 '
                                            '해결책은 현재의 높은 만성 질환의 사망률에 초점을 맞추었고, 주로 선진국에서 이용하고 '
                                            '있다. 개발도상국에서는 1차 원격 진단 방문으로 훨씬 더 잘 알려져 있는 새로운 방법이 '
                                            '최근에 생겨나고 있다. 그것에 따라 의사들은 원격으로 진찰하고 환자를 치료하기 위해서 '
                                            '기기를 사용한다. 예를 들어, 남아프리카가 있다. 1차 원격 진단 상담이 이미 진단 '
                                            '했었던 만성 질환을 감시할 뿐만 아니라, 미래에 발생하여 의사를 찾게 될 병까지도 '
                                            '진단하고 관리하기 때문에, 이 새로운 기술과 의료 원칙은 주요한 헬스 케어 제공 문제를 '
                                            '향상시키기 위한 중요한 약속을 담고 있다.',
                                 'qas': [{'answers': [{'answer_start': 132, 'text': '선진국'}],
                                          'id': '6526552-3-0',
                                          'question': '원격 모니터링 해결책은 높은 만성 질환의 사망률에 초점을 맞추었는데 '
                                                      '주로 이용하는 곳은?'},
                                         {'answers': [{'answer_start': 6,
                                                       'text': '환자들의 모니터링'}],
                                          'id': '6526552-3-1',
                                          'question': '혈압계 등을 사용하여 의료진에게 정보를 전송하는 최근에 생겨난 '
                                                      '서비스는?'},
                                         {'answers': [{'answer_start': 87,
                                                       'text': '원격 모니터링 해결책'}],
                                          'id': '6542182-3-0',
                                          'question': '주로 선진국에서 사용하며 높은 만성질환의 사망률에 초점을 맞춘 원격 '
                                                      '의료는? '},
                                         {'answers': [{'answer_start': 147, 'text': '개발도상국'}],
                                          'id': '6542182-3-1',
                                          'question': '1차 원격 진단 방문으로 잘 알려져 있는 새로운 원격 의료 방법이 '
                                                      '생겨나고 있는 곳은?'},
                                         {'answers': [{'answer_start': 156,
                                                       'text': '1차 원격 진단 방문'}],
                                          'id': '5847018-3-0',
                                          'question': '개발도상국에서 최근 생겨나고 있는 원격 의료의 새로운 방법은?'},
                                         {'answers': [{'answer_start': 87,
                                                       'text': '원격 모니터링 해결책'}],
                                          'id': '5847018-3-1',
                                          'question': '높은 만성 질환의 사망률에 초점을 맞춘, 주로 선진국에서 이용하는 '
                                                      '원격 의료 방식은?'}]},
                                {'context': '2014년 1월에, 오스트레일리아에서 멜버른 기술 전문 대학교는 호주 수유 협회()와 '
                                            '공동으로 최초로 새어머니들을 위한 손을 쓰지 않고 모유 수유를 할 수 있는 구글 글래스 '
                                            '애플리케이션을 만들기 위해서 스몰 월드 소셜(Small World Social)을 '
                                            '설립하였다. ‘구글 글래스 앱 평가판’ 애플리케이션으로 흔한 모유 수유에 관한 '
                                            '문제들(아기의 의도를 이해하는 것, 자세 등)에 관한 설명문을 읽으면서 아기를 보육할 '
                                            '있고, 혹은 안전한 구글 행아웃을 통하여 수유 상담가를 부를 수 있다. 수유 상담가는 '
                                            '엄마의 구글 글래스 카메라를 통해 문제를 볼 수 있다. 앱의 평가판은 2014년 4월에 '
                                            '멜버른에서 성공적으로 끝이 났고, 앱의 사용자 전원이 모두 자신감 있게 모유 수유를 할 '
                                            '수 있었다.',
                                 'qas': [{'answers': [{'answer_start': 113,
                                                       'text': '스몰 월드 소셜'}],
                                          'id': '6526552-4-0',
                                          'question': '새어머니들이 손을 쓰지 않고 모유 수유를 할 수 있게 만든 '
                                                      '애플리케이션 회사는?'},
                                         {'answers': [{'answer_start': 21,
                                                       'text': '멜버른 기술 전문 대학교'}],
                                          'id': '6526552-4-1',
                                          'question': '모유 수유 구글 글래스 애플리케이션을 만든 오스트레일리아 대학 '
                                                      '이름은?'},
                                         {'answers': [{'answer_start': 21,
                                                       'text': '멜버른 기술 전문 대학교'}],
                                          'id': '6542182-4-0',
                                          'question': '손을 쓰지 않고 모유 수유를 할 수는 애플리케이션을 만들기 위해 호주 '
                                                      '수유협회와 손을 잡은 대학은?'},
                                         {'answers': [{'answer_start': 113,
                                                       'text': '스몰 월드 소셜'}],
                                          'id': '6542182-4-1',
                                          'question': '멜버른 기술 전문 대학교가 구글 글래스 애플리케이션을 만들기 위해 '
                                                      '설립한 것은?'},
                                         {'answers': [{'answer_start': 322, 'text': '2014년'}],
                                          'id': '6542182-4-2',
                                          'question': '멜버른에서 구글 글래스 앱 평가판이 성공적으로 끝난 해는?'},
                                         {'answers': [{'answer_start': 113,
                                                       'text': '스몰 월드 소셜(Small World '
                                                               'Social)'}],
                                          'id': '5847018-4-0',
                                          'question': '멜버른 기술 전문 대학교와 호주 수유 협회가 모유 수유 어플리케이션을 '
                                                      '만들기 위해 설립한 회사의 이름은?'},
                                         {'answers': [{'answer_start': 287,
                                                       'text': '구글 글래스 카메라'}],
                                          'id': '5847018-4-1',
                                          'question': '수유 상담가는 무엇을 통하여 수유 문제를 겪는 엄마의 문제를 볼 수 '
                                                      '있는가?'}]}],
                 'title': '원격_의료'},
                {'paragraphs': [{'context': '그 조사 도중 부하 몇 사람이 해변에서 캠프를 하는 동안 티에라델푸에고 섬의 원주민이 '
                                            '배를 가지고 사라 졌다. 피츠로이는 그 배를 추적하고 난투를 벌인 후 범인의 가족을 '
                                            '인질로 잡아 배로 데리고 왔다. 결국 피츠로이는 가족들 중 두 남자와 각각 한 명의 '
                                            '소년, 소녀를 데리고 가게 되었다. 그들을 해안에 쉽게 내려주기엔 무리였으므로, '
                                            '피츠로이는 그 원주민들에게 영어와 기독교를 전파하면서 문명화시키기로 결심했다. 그들은 '
                                            '이후 세례명을 받았는데, 소녀는 ‘푸에지아 바스켓’, 소년은 ‘제임스 버튼’, 성인 '
                                            '남자의 한 명은 ‘요크 민스터’라고 이름을 붙여 주었다. 또 한 명의 소년은 ‘보트 '
                                            '메모리’라고 불렀는데, 잉글랜드로 돌아가 천연두 백신을 맞은 후에 사망했다. 그들은 '
                                            '선교사 리처드 매튜스의 주목을 받았고, 1831년 여름에는 국왕을 알현하는데 충분할 '
                                            '정도로 영국 문화에 익숙해졌다.',
                                 'qas': [{'answers': [{'answer_start': 32, 'text': '티에라델푸에고'}],
                                          'id': '6307345-0-0',
                                          'question': '배를 가지고 사라진 원주민이 사는 섬의 이름은?'},
                                         {'answers': [{'answer_start': 206, 'text': '기독교'}],
                                          'id': '6307345-0-1',
                                          'question': '피츠로이가 원주민들에게 전파하려는 종교의 이름은?'},
                                         {'answers': [{'answer_start': 380,
                                                       'text': '리처드 매튜스'}],
                                          'id': '6307345-0-2',
                                          'question': '피츠로이에게 세례명을 받은 원주민들을 주목한 선교사의 이름은?'},
                                         {'answers': [{'answer_start': 32, 'text': '티에라델푸에고'}],
                                          'id': '6496108-0-0',
                                          'question': '배를 잊어버린 섬의 이름은?'},
                                         {'answers': [{'answer_start': 195, 'text': '원주민'}],
                                          'id': '6496108-0-1',
                                          'question': '피츠로이가 문명화하려 했던 대상은?'},
                                         {'answers': [{'answer_start': 91, 'text': '가족'}],
                                          'id': '6496180-0-0',
                                          'question': '피츠로이가 배를 가져간 원주민과의 난투 끝에 인질로 잡아온 이는?'},
                                         {'answers': [{'answer_start': 202,
                                                       'text': '영어와 기독교'}],
                                          'id': '6496180-0-1',
                                          'question': '피츠로이는 인질로 잡은 원주민들에게 무엇을 전파하였나?'},
                                         {'answers': [{'answer_start': 326, 'text': '보트 메모리'}],
                                          'id': '6496180-0-2',
                                          'question': '인질로 잡힌 원주민들 중 천연두 백신 접종 후 사망한 소년의 '
                                                      '이름은?'}]},
                                {'context': '1832년 3월에 브라질 바이아의 기억은 인상적이었다. 다윈은 노예들을 취급한 이야기에 '
                                            '질렸지만, 피츠로이는 잔인한 행위를 지지하지는 않았지만, 일찍이 농장주가 노예에게 '
                                            "해방시켜 줄까라고 물었을 때에 노예는 '아니오'라고 대답했다고 이야기했다. 다윈은 "
                                            '솔직하게, "노예가 주인의 앞에서 그런 질문에 정직하게 대답할까요?"라고 피츠로이에게 '
                                            '물었다. 피츠로이는 발끈하며, 만약 자신의 말을 의심한다면 더 이상 함께 갈 수 없다고 '
                                            '화를 냈다. 피츠로이는 다윈을 함장의 테이블에서 쫓아냈지만, 이후 다윈에게 발끈한 것을 '
                                            '솔직하게 사죄했다. 그 이후 노예 문제에 대한 주제는 서로 회피했다. 그러나 그들의 '
                                            '종교적, 이념적 다툼은 끝나지 않았고, 그러한 불화는 항해가 끝난 후 닥쳐오게 된다.',
                                 'qas': [{'answers': [{'answer_start': 117, 'text': '아니오'}],
                                          'id': '6307345-1-0',
                                          'question': '농장주가 노예에게 해방을 원하냐고 물어봤을때 노예의 대답은?'},
                                         {'answers': [{'answer_start': 31, 'text': '다윈'}],
                                          'id': '6307345-1-1',
                                          'question': '노예문제 등 이념적 종교적으로 피츠로이와 다퉜던 사람의 이름은?'},
                                         {'answers': [{'answer_start': 10, 'text': '브라질'}],
                                          'id': '6307345-1-2',
                                          'question': '다윈과 피츠로이가 서로 노예 문제에 대한 이야기를 한 나라의 '
                                                      '이름은?'},
                                         {'answers': [{'answer_start': 31, 'text': '다윈'}],
                                          'id': '6496180-1-0',
                                          'question': '노예 문제로 피츠로이와 갈등했던 인물은? '},
                                         {'answers': [{'answer_start': 255,
                                                       'text': '함장의 테이블'}],
                                          'id': '6496180-1-1',
                                          'question': '피츠로이는 자신의 말을 의심한 다윈을 어디에서 쫓아냈나?'},
                                         {'answers': [{'answer_start': 364, 'text': '항해'}],
                                          'id': '6496180-1-2',
                                          'question': '피츠로이와 다윈의 불화가 닥쳐온 것은 무엇이 끝난 뒤였나?'}]},
                                {'context': '피츠로이는 그때 전 함장의 메모도 함께 항해 보고서로 작성했다. 이것은 필립 파커 킹 '
                                            '함장, 다윈의 기록과 함께 3권의 《영국 군함 어드벤처호와 비글호의 조사 항해 '
                                            '기록》으로 1839년에 간행되었다. 피츠로이의 보고는 구약성서 창세기의 대홍수 이야기에 '
                                            '관해서는 "성서의 신빙성을 부정하는 지질 학자"라는 연구평가 의견이 있었다. 또한 '
                                            '광대한 대지가 40일간의 홍수에 퇴적할 수 없는 것을 인정하고, 성서의 설명을 믿지 '
                                            '않는다는 생각을 보여주었다. 그 때 그는 그런 생각이 "젊은 수병의 눈에 닿는" 것을 '
                                            '우려하여 포탄을 포함한 바위 산 정상 부근에 쌓여있는 것은 노아의 대홍수 고백이라는 '
                                            '논의로 성서의 문자해석에 대한 설명을 하고 있다. 피츠로이는 항해를 하는 동안 '
                                            '받아들이고 있었던 라이엘의 새로운 아이디어로부터 자신을 분리하기 시작했다. 그리고 아주 '
                                            '신앙심이 깊었던 아내의 영향도 있어, 성공회의 전통적인 교리를 받아들여갔다.',
                                 'qas': [{'answers': [{'answer_start': 40, 'text': '필립 파커 킹'}],
                                          'id': '6307345-2-0',
                                          'question': '피츠로이의 전 함장의 이름은?'},
                                         {'answers': [{'answer_start': 98, 'text': '1839년'}],
                                          'id': '6307345-2-1',
                                          'question': '영국 군함 어드벤처호와 비글호의 조사 항해 기록이 발간된 해는?'},
                                         {'answers': [{'answer_start': 443, 'text': '성공회'}],
                                          'id': '6307345-2-2',
                                          'question': '피츠로이가 아내의 영향으로 인해 받아들인 종교의 이름은?'},
                                         {'answers': [{'answer_start': 98, 'text': '1839년'}],
                                          'id': '6496180-2-0',
                                          'question': '필립 파커 킹 함장과 다윈 등의 기록물이 책으로 간행된 연도는?'},
                                         {'answers': [{'answer_start': 431, 'text': '아내'}],
                                          'id': '6496180-2-1',
                                          'question': '피츠로이가 성공회의 전통적 교리를 수용하는데 영향을 끼친 사람은?'},
                                         {'answers': [{'answer_start': 383, 'text': '라이엘'}],
                                          'id': '6496180-2-2',
                                          'question': '항해하는 동안 피츠로이는 누구의 생각으로부터 자신을 분리하기 '
                                                      '시작했나?'}]}],
                 'title': '로버트_피츠로이'}]

  def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
      return True
    return False

  examples = []
  for entry in input_data:
    for paragraph in entry["paragraphs"]:
      paragraph_text = paragraph["context"]
      doc_tokens = []
      char_to_word_offset = []
      prev_is_whitespace = True
      for c in paragraph_text:
        if is_whitespace(c):
          prev_is_whitespace = True
        else:
          if prev_is_whitespace:
            doc_tokens.append(c)
          else:
            doc_tokens[-1] += c
          prev_is_whitespace = False
        #지문데이터의 length에서 -1을 해준값이 들어가있는 리스트 데이터
        char_to_word_offset.append(len(doc_tokens) - 1)

      for qa in paragraph["qas"]:
        qas_id = qa["id"]
        question_text = qa["question"]
        start_position = None
        end_position = None
        orig_answer_text = None
        is_impossible = False
        if is_training:

          if FLAGS.version_2_with_negative:
            is_impossible = qa["is_impossible"]
          if (len(qa["answers"]) != 1) and (not is_impossible):
            raise ValueError(
                "For training, each question should have exactly 1 answer.")
          if not is_impossible:
            answer = qa["answers"][0]
            orig_answer_text = answer["text"]
            answer_offset = answer["answer_start"]
            answer_length = len(orig_answer_text)
            start_position = char_to_word_offset[answer_offset]
            end_position = char_to_word_offset[answer_offset + answer_length -
                                               1]
            # Only add answers where the text can be exactly recovered from the
            # document. If this CAN'T happen it's likely due to weird Unicode
            # stuff so we will just skip the example.
            #
            # Note that this means for training mode, every example is NOT
            # guaranteed to be preserved.
            actual_text = " ".join(
                doc_tokens[start_position:(end_position + 1)])
            cleaned_answer_text = " ".join(
                tokenization.whitespace_tokenize(orig_answer_text))
            if actual_text.find(cleaned_answer_text) == -1:
              tf.logging.warning("Could not find answer: '%s' vs. '%s'",
                                 actual_text, cleaned_answer_text)
              continue
          else:
            start_position = -1
            end_position = -1
            orig_answer_text = ""

        example = SquadExample(
            qas_id=qas_id,
            question_text=question_text,
            doc_tokens=doc_tokens,
            orig_answer_text=orig_answer_text,
            start_position=start_position,
            end_position=end_position,
            is_impossible=is_impossible)
        examples.append(example)


  return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 output_fn):
  """Loads a data file into a list of `InputBatch`s."""

  unique_id = 1000000000

  for (example_index, example) in enumerate(examples):
    query_tokens = tokenizer.tokenize(example.question_text)

    if len(query_tokens) > max_query_length:
      query_tokens = query_tokens[0:max_query_length]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
      orig_to_tok_index.append(len(all_doc_tokens))
      sub_tokens = tokenizer.tokenize(token)
      for sub_token in sub_tokens:
        tok_to_orig_index.append(i)
        all_doc_tokens.append(sub_token)

    tok_start_position = None
    tok_end_position = None
    if is_training and example.is_impossible:
      tok_start_position = -1
      tok_end_position = -1
    if is_training and not example.is_impossible:
      tok_start_position = orig_to_tok_index[example.start_position]
      if example.end_position < len(example.doc_tokens) - 1:
        tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
      else:
        tok_end_position = len(all_doc_tokens) - 1
      (tok_start_position, tok_end_position) = _improve_answer_span(
          all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
          example.orig_answer_text)

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
      length = len(all_doc_tokens) - start_offset
      if length > max_tokens_for_doc:
        length = max_tokens_for_doc
      doc_spans.append(_DocSpan(start=start_offset, length=length))
      if start_offset + length == len(all_doc_tokens):
        break
      start_offset += min(length, doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
      tokens = []
      token_to_orig_map = {}
      token_is_max_context = {}
      segment_ids = []
      tokens.append("[CLS]")
      segment_ids.append(0)
      for token in query_tokens:
        tokens.append(token)
        segment_ids.append(0)
      tokens.append("[SEP]")
      segment_ids.append(0)

      for i in range(doc_span.length):
        split_token_index = doc_span.start + i
        token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

        is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                               split_token_index)
        token_is_max_context[len(tokens)] = is_max_context
        tokens.append(all_doc_tokens[split_token_index])
        segment_ids.append(1)
      tokens.append("[SEP]")
      segment_ids.append(1)

      input_ids = tokenizer.convert_tokens_to_ids(tokens)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      input_mask = [1] * len(input_ids)

      # Zero-pad up to the sequence length.
      while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(segment_ids) == max_seq_length

      start_position = None
      end_position = None
      if is_training and not example.is_impossible:
        # For training, if our document chunk does not contain an annotation
        # we throw it out, since there is nothing to predict.
        doc_start = doc_span.start
        doc_end = doc_span.start + doc_span.length - 1
        out_of_span = False
        if not (tok_start_position >= doc_start and
                tok_end_position <= doc_end):
          out_of_span = True
        if out_of_span:
          start_position = 0
          end_position = 0
        else:
          doc_offset = len(query_tokens) + 2
          start_position = tok_start_position - doc_start + doc_offset
          end_position = tok_end_position - doc_start + doc_offset

      if is_training and example.is_impossible:
        start_position = 0
        end_position = 0

      if example_index < 20:
        tf.logging.info("*** Example ***")
        tf.logging.info("unique_id: %s" % (unique_id))
        tf.logging.info("example_index: %s" % (example_index))
        tf.logging.info("doc_span_index: %s" % (doc_span_index))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("token_to_orig_map: %s" % " ".join(
            ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
        tf.logging.info("token_is_max_context: %s" % " ".join([
            "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
        ]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info(
            "input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info(
            "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        if is_training and example.is_impossible:
          tf.logging.info("impossible example")
        if is_training and not example.is_impossible:
          answer_text = " ".join(tokens[start_position:(end_position + 1)])
          tf.logging.info("start_position: %d" % (start_position))
          tf.logging.info("end_position: %d" % (end_position))
          tf.logging.info(
              "answer: %s" % (tokenization.printable_text(answer_text)))

      feature = InputFeatures(
          unique_id=unique_id,
          example_index=example_index,
          doc_span_index=doc_span_index,
          tokens=tokens,
          token_to_orig_map=token_to_orig_map,
          token_is_max_context=token_is_max_context,
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids,
          start_position=start_position,
          end_position=end_position,
          is_impossible=example.is_impossible)

      # Run callback
      output_fn(feature)

      unique_id += 1


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
  """Returns tokenized answer spans that better match the annotated answer."""

  # The SQuAD annotations are character based. We first project them to
  # whitespace-tokenized words. But then after WordPiece tokenization, we can
  # often find a "better match". For example:
  #
  #   Question: What year was John Smith born?
  #   Context: The leader was John Smith (1895-1943).
  #   Answer: 1895
  #
  # The original whitespace-tokenized answer will be "(1895-1943).". However
  # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
  # the exact answer, 1895.
  #
  # However, this is not always possible. Consider the following:
  #
  #   Question: What country is the top exporter of electornics?
  #   Context: The Japanese electronics industry is the lagest in the world.
  #   Answer: Japan
  #
  # In this case, the annotator chose "Japan" as a character sub-span of
  # the word "Japanese". Since our WordPiece tokenizer does not split
  # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
  # in SQuAD, but does happen.
  tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

  for new_start in range(input_start, input_end + 1):
    for new_end in range(input_end, new_start - 1, -1):
      text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
      if text_span == tok_answer_text:
        return (new_start, new_end)

  return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""

  # Because of the sliding window approach taken to scoring documents, a single
  # token can appear in multiple documents. E.g.
  #  Doc: the man went to the store and bought a gallon of milk
  #  Span A: the man went to the
  #  Span B: to the store and bought
  #  Span C: and bought a gallon of
  #  ...
  #
  # Now the word 'bought' will have two scores from spans B and C. We only
  # want to consider the score with "maximum context", which we define as
  # the *minimum* of its left and right context (the *sum* of left and
  # right context will always be the same, of course).
  #
  # In the example the maximum context for 'bought' would be span C since
  # it has 1 left context and 3 right context, while span B has 4 left context
  # and 0 right context.
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)
  #이값을 지지고 볶고 해서 start_logits end_logits를 만든다
  #final hidden layer를 가져온다
  final_hidden = model.get_sequence_output()

  print(final_hidden)
  print('dd')

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  print(sess.run(final_hidden))
  import sys
  sys.exit(1)

  final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
  batch_size = final_hidden_shape[0]
  seq_length = final_hidden_shape[1]
  hidden_size = final_hidden_shape[2]

  print('batch size')
  print(batch_size)
  print(seq_length)
  print('seq_length')


  output_weights = tf.get_variable(
      "cls/squad/output_weights", [2, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "cls/squad/output_bias", [2], initializer=tf.zeros_initializer())

  #아마 가중치값(output_weights)와 행렬곱(matmul)을 하려고 reshape를 하는듯 하다
  #final_hidden = bert에 input값을 넣었을때 생성되는 마지막 hidden layer
  final_hidden_matrix = tf.reshape(final_hidden,
                                   [batch_size * seq_length, hidden_size])

  #question anwsering task를 수행하는 layer#
  ###################fully connected layer####################
  logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)
  logits = tf.reshape(logits, [batch_size, seq_length, 2])
  logits = tf.transpose(logits, [2, 0, 1])

  unstacked_logits = tf.unstack(logits, axis=0)

  (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])
  ###################fully connected layer####################
  return (start_logits, end_logits)


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    #https://github.com/google-research/bert/issues/177 관련 글
    (start_logits, end_logits) = create_model(
        bert_config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      seq_length = modeling.get_shape_list(input_ids)[1]

      def compute_loss(logits, positions):
        one_hot_positions = tf.one_hot(
            positions, depth=seq_length, dtype=tf.float32)
        #softmax는 값들이 0과 1사이로 나오는 반면에 log softmax는 -infinity 에서 0사이의 값이 나온다
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        loss = -tf.reduce_mean(
            tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
        # print("ㅎㅇㅎㅇㅎㅇㅎㅇ")
        # sess = tf.InteractiveSession()
        # init_op = tf.initialize_all_variables()
        # sess.run(init_op)
        # sess.run(loss)
        return loss

      start_positions = features["start_positions"]
      end_positions = features["end_positions"]

      start_loss = compute_loss(start_logits, start_positions)
      end_loss = compute_loss(end_logits, end_positions)

      total_loss = (start_loss + end_loss) / 2.0




      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
          "unique_ids": unique_ids,
          "start_logits": start_logits,
          "end_logits": end_logits,
      }
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    else:
      raise ValueError(
          "Only TRAIN and PREDICT modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "unique_ids": tf.FixedLenFeature([], tf.int64),
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
  }

  if is_training:
    name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
    name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file):
  #여기 보면됨 2020/01/11
  """Write final predictions to the json file and log-odds of null if needed."""
  tf.logging.info("Writing predictions to: %s" % (output_prediction_file))
  tf.logging.info("Writing nbest to: %s" % (output_nbest_file))

  example_index_to_features = collections.defaultdict(list)
  for feature in all_features:
    example_index_to_features[feature.example_index].append(feature)

  unique_id_to_result = {}
  for result in all_results:
    unique_id_to_result[result.unique_id] = result

  _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
      "PrelimPrediction",
      ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

  all_predictions = collections.OrderedDict()
  all_nbest_json = collections.OrderedDict()
  scores_diff_json = collections.OrderedDict()

  for (example_index, example) in enumerate(all_examples):
    features = example_index_to_features[example_index]

    prelim_predictions = []
    # keep track of the minimum score of null start+end of position 0
    score_null = 1000000  # large and positive
    min_null_feature_index = 0  # the paragraph slice with min mull score
    null_start_logit = 0  # the start logit at the slice with min null score
    null_end_logit = 0  # the end logit at the slice with min null score
    for (feature_index, feature) in enumerate(features):
      result = unique_id_to_result[feature.unique_id]
      start_indexes = _get_best_indexes(result.start_logits, n_best_size)
      end_indexes = _get_best_indexes(result.end_logits, n_best_size)
      # if we could have irrelevant answers, get the min score of irrelevant
      if FLAGS.version_2_with_negative:
        feature_null_score = result.start_logits[0] + result.end_logits[0]
        if feature_null_score < score_null:
          score_null = feature_null_score
          min_null_feature_index = feature_index
          null_start_logit = result.start_logits[0]
          null_end_logit = result.end_logits[0]
      for start_index in start_indexes:
        for end_index in end_indexes:
          # We could hypothetically create invalid predictions, e.g., predict
          # that the start of the span is in the question. We throw out all
          # invalid predictions.
          if start_index >= len(feature.tokens):
            continue
          if end_index >= len(feature.tokens):
            continue
          if start_index not in feature.token_to_orig_map:
            continue
          if end_index not in feature.token_to_orig_map:
            continue
          if not feature.token_is_max_context.get(start_index, False):
            continue
          if end_index < start_index:
            continue
          length = end_index - start_index + 1
          if length > max_answer_length:
            continue
          prelim_predictions.append(
              _PrelimPrediction(
                  feature_index=feature_index,
                  start_index=start_index,
                  end_index=end_index,
                  start_logit=result.start_logits[start_index],
                  end_logit=result.end_logits[end_index]))

    if FLAGS.version_2_with_negative:
      prelim_predictions.append(
          _PrelimPrediction(
              feature_index=min_null_feature_index,
              start_index=0,
              end_index=0,
              start_logit=null_start_logit,
              end_logit=null_end_logit))
    #관련 설명글 https://github.com/google-research/bert/issues/177
    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True)

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_logit", "end_logit"])

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
      if len(nbest) >= n_best_size:
        break
      feature = features[pred.feature_index]
      if pred.start_index > 0:  # this is a non-null prediction
        tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]

        #paragraph/context에서 정답의 시작지점과 끝점을 알 수 있다
        orig_doc_start = feature.token_to_orig_map[pred.start_index]
        orig_doc_end = feature.token_to_orig_map[pred.end_index]

        orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
        tok_text = " ".join(tok_tokens)

        # De-tokenize WordPieces that have been split off.
        tok_text = tok_text.replace(" ##", "")
        tok_text = tok_text.replace("##", "")

        # Clean whitespace
        tok_text = tok_text.strip()
        tok_text = " ".join(tok_text.split())
        orig_text = " ".join(orig_tokens)

        final_text = get_final_text(tok_text, orig_text, do_lower_case)
        if final_text in seen_predictions:
          continue

        seen_predictions[final_text] = True
      else:
        final_text = ""
        seen_predictions[final_text] = True

      nbest.append(
          _NbestPrediction(
              text=final_text,
              start_logit=pred.start_logit,
              end_logit=pred.end_logit))

    # if we didn't inlude the empty option in the n-best, inlcude it
    if FLAGS.version_2_with_negative:
      if "" not in seen_predictions:
        nbest.append(
            _NbestPrediction(
                text="", start_logit=null_start_logit,
                end_logit=null_end_logit))
    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
      nbest.append(
          _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

    assert len(nbest) >= 1

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
      total_scores.append(entry.start_logit + entry.end_logit)
      if not best_non_null_entry:
        if entry.text:
          best_non_null_entry = entry

    probs = _compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):
      output = collections.OrderedDict()
      output["text"] = entry.text
      output["probability"] = probs[i]
      output["start_logit"] = entry.start_logit
      output["end_logit"] = entry.end_logit
      nbest_json.append(output)

    assert len(nbest_json) >= 1

    if not FLAGS.version_2_with_negative:
      all_predictions[example.qas_id] = nbest_json[0]["text"]
    else:
      # predict "" iff the null score - the score of best non-null > threshold
      score_diff = score_null - best_non_null_entry.start_logit - (
          best_non_null_entry.end_logit)
      scores_diff_json[example.qas_id] = score_diff
      if score_diff > FLAGS.null_score_diff_threshold:
        all_predictions[example.qas_id] = ""
      else:
        all_predictions[example.qas_id] = best_non_null_entry.text

    all_nbest_json[example.qas_id] = nbest_json

  with tf.gfile.GFile(output_prediction_file, "w") as writer:
    writer.write(json.dumps(all_predictions, indent=4) + "\n")

  with tf.gfile.GFile(output_nbest_file, "w") as writer:
    writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

  if FLAGS.version_2_with_negative:
    with tf.gfile.GFile(output_null_log_odds_file, "w") as writer:
      writer.write(json.dumps(scores_diff_json, indent=4) + "\n")


def get_final_text(pred_text, orig_text, do_lower_case):
  """Project the tokenized prediction back to the original text."""

  # When we created the data, we kept track of the alignment between original
  # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
  # now `orig_text` contains the span of our original text corresponding to the
  # span that we predicted.
  #
  # However, `orig_text` may contain extra characters that we don't want in
  # our prediction.
  #
  # For example, let's say:
  #   pred_text = steve smith
  #   orig_text = Steve Smith's
  #
  # We don't want to return `orig_text` because it contains the extra "'s".
  #
  # We don't want to return `pred_text` because it's already been normalized
  # (the SQuAD eval script also does punctuation stripping/lower casing but
  # our tokenizer does additional normalization like stripping accent
  # characters).
  #
  # What we really want to return is "Steve Smith".
  #
  # Therefore, we have to apply a semi-complicated alignment heruistic between
  # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
  # can fail in certain cases in which case we just return `orig_text`.

  def _strip_spaces(text):
    ns_chars = []
    ns_to_s_map = collections.OrderedDict()
    for (i, c) in enumerate(text):
      if c == " ":
        continue
      ns_to_s_map[len(ns_chars)] = i
      ns_chars.append(c)
    ns_text = "".join(ns_chars)
    return (ns_text, ns_to_s_map)

  # We first tokenize `orig_text`, strip whitespace from the result
  # and `pred_text`, and check if they are the same length. If they are
  # NOT the same length, the heuristic has failed. If they are the same
  # length, we assume the characters are one-to-one aligned.
  tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

  tok_text = " ".join(tokenizer.tokenize(orig_text))

  start_position = tok_text.find(pred_text)
  if start_position == -1:
    if FLAGS.verbose_logging:
      tf.logging.info(
          "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
    return orig_text
  end_position = start_position + len(pred_text) - 1

  (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
  (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

  if len(orig_ns_text) != len(tok_ns_text):
    if FLAGS.verbose_logging:
      tf.logging.info("Length not equal after stripping spaces: '%s' vs '%s'",
                      orig_ns_text, tok_ns_text)
    return orig_text

  # We then project the characters in `pred_text` back to `orig_text` using
  # the character-to-character alignment.
  tok_s_to_ns_map = {}
  for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
    tok_s_to_ns_map[tok_index] = i

  orig_start_position = None
  if start_position in tok_s_to_ns_map:
    ns_start_position = tok_s_to_ns_map[start_position]
    if ns_start_position in orig_ns_to_s_map:
      orig_start_position = orig_ns_to_s_map[ns_start_position]

  if orig_start_position is None:
    if FLAGS.verbose_logging:
      tf.logging.info("Couldn't map start position")
    return orig_text

  orig_end_position = None
  if end_position in tok_s_to_ns_map:
    ns_end_position = tok_s_to_ns_map[end_position]
    if ns_end_position in orig_ns_to_s_map:
      orig_end_position = orig_ns_to_s_map[ns_end_position]

  if orig_end_position is None:
    if FLAGS.verbose_logging:
      tf.logging.info("Couldn't map end position")
    return orig_text

  output_text = orig_text[orig_start_position:(orig_end_position + 1)]
  return output_text


def _get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes


def _compute_softmax(scores):
  """Compute softmax probability over raw logits."""
  if not scores:
    return []

  max_score = None
  for score in scores:
    if max_score is None or score > max_score:
      max_score = score

  exp_scores = []
  total_sum = 0.0
  for score in scores:
    x = math.exp(score - max_score)
    exp_scores.append(x)
    total_sum += x

  probs = []
  for score in exp_scores:
    probs.append(score / total_sum)
  return probs


class FeatureWriter(object):
  """Writes InputFeature to TF example file."""

  def __init__(self, filename, is_training):
    self.filename = filename
    self.is_training = is_training
    self.num_features = 0
    self._writer = tf.python_io.TFRecordWriter(filename)

  def process_feature(self, feature):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
    self.num_features += 1

    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
      return feature

    features = collections.OrderedDict()
    features["unique_ids"] = create_int_feature([feature.unique_id])
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)

    if self.is_training:
      features["start_positions"] = create_int_feature([feature.start_position])
      features["end_positions"] = create_int_feature([feature.end_position])
      impossible = 0
      if feature.is_impossible:
        impossible = 1
      features["is_impossible"] = create_int_feature([impossible])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())

  def close(self):
    self._writer.close()


def validate_flags_or_throw(bert_config):
  """Validate the input FLAGS or throw an exception."""
  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_predict:
    raise ValueError("At least one of `do_train` or `do_predict` must be True.")

  if FLAGS.do_train:
    if not FLAGS.train_file:
      raise ValueError(
          "If `do_train` is True, then `train_file` must be specified.")
  if FLAGS.do_predict:
    if not FLAGS.predict_file:
      raise ValueError(
          "If `do_predict` is True, then `predict_file` must be specified.")

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
    raise ValueError(
        "The max_seq_length (%d) must be greater than max_query_length "
        "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))


def main(_):
  #tensorflow = 1.15.0 버전을 사용할때 코드가 잘돌아간다

  tf.logging.set_verbosity(tf.logging.INFO)
  #hidden size는 어떤것으로하고 활성화 함수는 어떤것을 사용하는지에 대한 설정 json파일
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  validate_flags_or_throw(bert_config)

  #gfile은 tensorflow 용 파일 입출력 함수라고 한다 https://ballentain.tistory.com/21
  tf.gfile.MakeDirs(FLAGS.output_dir)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  #tpu를 사용할때 쓰는 코드들#
  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))
  #tpu를 사용할때 쓰는 코드들#

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None

  # 학습을 시킨다는 설정 값을 줬을때 발동되는 조건문
  if FLAGS.do_train:
      # train_examples에 학습데이터(json 파일)이 파싱되어서 리스트 형태로 들어간다. 아래에는 리스트안에 들어가는 값들을 적어놨다
      # qas_id,
      # question_text
      # doc_tokens 지문 데이터
      # start_position
      # end_position
      # is_impossible
    # train_examples = read_squad_examples( 수정
    #     input_file=FLAGS.train_file, is_training=True) #is training이 false일 경우 start & end position이 없어진다 수정
    train_examples = read_squad_examples(is_training=True)  # is training이 false일 경우 start & end position이 없어진다

  #모르겠다
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    #모르겠다
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    # Pre-shuffle the input to avoid having to make a very large shuffle
    # buffer in in the `input_fn`.
    # 리스트에 있는 원소들을 섞는다
    rng = random.Random(12345)
    rng.shuffle(train_examples)
    print(train_examples)
  print("zzhisdfsdfsd")

  #모델을 어떻게 학습시킬지에 대한 값들이 들어있다
  #2020/1/11 여기부터 보면된다
  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  #TPU와 관련된 설정값
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  print("zzhi")

  if FLAGS.do_train:
    # We write to a temporary file to avoid storing very large constant tensors
    # in memory.
    # 너무 큰 constant tesnor가 저장되지 않도록 임시 저장을 시켰다고 한다.
    train_writer = FeatureWriter(
        filename=os.path.join(FLAGS.output_dir, "train.tf_record"),
        is_training=True)
    #모르겠다
    convert_examples_to_features(
        examples=train_examples,
        tokenizer=tokenizer,
        max_seq_length=FLAGS.max_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_query_length=FLAGS.max_query_length,
        is_training=True,
        output_fn=train_writer.process_feature)
    train_writer.close()

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num orig examples = %d", len(train_examples))
    tf.logging.info("  Num split examples = %d", train_writer.num_features)
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    del train_examples

    #학습시키기 위한 input값을 설정하는듯 하다
    train_input_fn = input_fn_builder(
        input_file=train_writer.filename,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  #예측을 한다는 설정값을 정했을때 발동된다
  if FLAGS.do_predict:
    #dev set을 가져온다
    #eval은 evaluation의 약자인듯 하다.
    eval_examples = read_squad_examples(
        input_file=FLAGS.predict_file, is_training=False)

    eval_writer = FeatureWriter(
        filename=os.path.join(FLAGS.output_dir, "eval.tf_record"),
        is_training=False)
    eval_features = []

    def append_feature(feature):
      eval_features.append(feature)
      eval_writer.process_feature(feature)

    convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=FLAGS.max_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_query_length=FLAGS.max_query_length,
        is_training=False,
        output_fn=append_feature)
    eval_writer.close()

    tf.logging.info("***** Running predictions *****")
    tf.logging.info("  Num orig examples = %d", len(eval_examples))
    tf.logging.info("  Num split examples = %d", len(eval_features))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    all_results = []

    predict_input_fn = input_fn_builder(
        input_file=eval_writer.filename,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)

    # If running eval on the TPU, you will need to specify the number of
    # steps.
    # 예측한 결과를 보여주는 코드들
    all_results = []
    for result in estimator.predict(
        predict_input_fn, yield_single_examples=True):
      if len(all_results) % 1000 == 0:
        tf.logging.info("Processing example: %d" % (len(all_results)))
      unique_id = int(result["unique_ids"])
      start_logits = [float(x) for x in result["start_logits"].flat]
      end_logits = [float(x) for x in result["end_logits"].flat]
      all_results.append(
          RawResult(
              unique_id=unique_id,
              start_logits=start_logits,
              end_logits=end_logits))

    output_prediction_file = os.path.join(FLAGS.output_dir, "predictions.json")
    output_nbest_file = os.path.join(FLAGS.output_dir, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(FLAGS.output_dir, "null_odds.json")

    write_predictions(eval_examples, eval_features, all_results,
                      FLAGS.n_best_size, FLAGS.max_answer_length,
                      FLAGS.do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file)


if __name__ == "__main__":
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
