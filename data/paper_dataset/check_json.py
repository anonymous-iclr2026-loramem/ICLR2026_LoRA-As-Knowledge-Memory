#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON 파일 검증 및 분석 스크립트
paperQA.json 파일의 구조와 내용을 검증하고 분석합니다.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple


def validate_json_file(filepath: str) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """JSON 파일을 검증하고 데이터를 반환합니다."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            return False, "JSON의 최상위 구조는 배열이어야 합니다.", []
        
        return True, "JSON 파일이 유효합니다.", data
    
    except json.JSONDecodeError as e:
        return False, f"JSON 구문 오류: {e}", []
    except FileNotFoundError:
        return False, f"파일을 찾을 수 없습니다: {filepath}", []
    except Exception as e:
        return False, f"파일 읽기 오류: {e}", []


def analyze_paper_structure(papers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """논문 데이터의 구조를 분석합니다."""
    analysis = {
        "총_논문_수": len(papers),
        "컨퍼런스별_분포": {},
        "필수_필드_누락": [],
        "introduction_길이_분포": {},
        "유효하지_않은_데이터": []
    }
    
    required_fields = ["id", "link", "conference", "title", "introduction", "QA"]
    
    for i, paper in enumerate(papers):
        # 필수 필드 확인
        missing_fields = [field for field in required_fields if field not in paper]
        if missing_fields:
            analysis["필수_필드_누락"].append({
                "인덱스": i,
                "id": paper.get("id", "N/A"),
                "누락된_필드": missing_fields
            })
        
        # 컨퍼런스별 분포
        conference = paper.get("conference", "Unknown")
        analysis["컨퍼런스별_분포"][conference] = analysis["컨퍼런스별_분포"].get(conference, 0) + 1
        
        # Introduction 길이 분석
        introduction = paper.get("introduction", "")
        if isinstance(introduction, str):
            length_category = get_length_category(len(introduction))
            analysis["introduction_길이_분포"][length_category] = analysis["introduction_길이_분포"].get(length_category, 0) + 1
        
        # 데이터 유효성 검사
        if not isinstance(paper.get("id"), int):
            analysis["유효하지_않은_데이터"].append(f"인덱스 {i}: ID가 정수가 아님")
        
        if not isinstance(paper.get("QA"), list):
            analysis["유효하지_않은_데이터"].append(f"인덱스 {i}: QA가 배열이 아님")
    
    return analysis


def get_length_category(length: int) -> str:
    """텍스트 길이를 카테고리로 분류합니다."""
    if length < 1000:
        return "짧음 (<1K)"
    elif length < 3000:
        return "보통 (1K-3K)"
    elif length < 5000:
        return "긺 (3K-5K)"
    else:
        return "매우긺 (>5K)"


def check_data_consistency(papers: List[Dict[str, Any]]) -> List[str]:
    """데이터 일관성을 확인합니다."""
    issues = []
    
    # ID 중복 확인
    ids = [paper.get("id") for paper in papers if "id" in paper]
    if len(ids) != len(set(ids)):
        issues.append("중복된 ID가 존재합니다.")
    
    # ID 순서 확인
    expected_ids = list(range(len(papers)))
    actual_ids = [paper.get("id") for paper in papers]
    if actual_ids != expected_ids:
        issues.append("ID가 순차적이지 않습니다.")
    
    # URL 형식 확인
    for i, paper in enumerate(papers):
        link = paper.get("link", "")
        if not link.startswith("https://arxiv.org/abs/"):
            issues.append(f"인덱스 {i}: 잘못된 arXiv URL 형식")
    
    return issues


def print_analysis_report(analysis: Dict[str, Any], consistency_issues: List[str]):
    """분석 결과를 출력합니다."""
    print("=" * 50)
    print("📊 JSON 파일 분석 보고서")
    print("=" * 50)
    
    print(f"\n📋 기본 정보:")
    print(f"  • 총 논문 수: {analysis['총_논문_수']}")
    
    print(f"\n🏛️ 컨퍼런스별 분포:")
    for conference, count in sorted(analysis['컨퍼런스별_분포'].items()):
        percentage = (count / analysis['총_논문_수']) * 100
        print(f"  • {conference}: {count}편 ({percentage:.1f}%)")
    
    print(f"\n📝 Introduction 길이 분포:")
    for category, count in sorted(analysis['introduction_길이_분포'].items()):
        percentage = (count / analysis['총_논문_수']) * 100
        print(f"  • {category}: {count}편 ({percentage:.1f}%)")
    
    if analysis['필수_필드_누락']:
        print(f"\n❌ 필수 필드 누락 ({len(analysis['필수_필드_누락'])}건):")
        for issue in analysis['필수_필드_누락'][:5]:  # 최대 5개만 표시
            print(f"  • 인덱스 {issue['인덱스']} (ID: {issue['id']}): {', '.join(issue['누락된_필드'])}")
        if len(analysis['필수_필드_누락']) > 5:
            print(f"  • ... 총 {len(analysis['필수_필드_누락'])}건")
    
    if analysis['유효하지_않은_데이터']:
        print(f"\n⚠️ 유효하지 않은 데이터 ({len(analysis['유효하지_않은_데이터'])}건):")
        for issue in analysis['유효하지_않은_데이터'][:5]:
            print(f"  • {issue}")
        if len(analysis['유효하지_않은_데이터']) > 5:
            print(f"  • ... 총 {len(analysis['유효하지_않은_데이터'])}건")
    
    if consistency_issues:
        print(f"\n🔍 데이터 일관성 문제 ({len(consistency_issues)}건):")
        for issue in consistency_issues:
            print(f"  • {issue}")
    
    if not analysis['필수_필드_누락'] and not analysis['유효하지_않은_데이터'] and not consistency_issues:
        print(f"\n✅ 모든 검사를 통과했습니다!")


def show_sample_papers(papers: List[Dict[str, Any]], count: int = 3):
    """샘플 논문 정보를 표시합니다."""
    print(f"\n📄 샘플 논문 정보 (처음 {min(count, len(papers))}편):")
    print("-" * 50)
    
    for i, paper in enumerate(papers[:count]):
        print(f"\n[{i+1}] ID: {paper.get('id', 'N/A')}")
        print(f"    제목: {paper.get('title', 'N/A')[:80]}{'...' if len(paper.get('title', '')) > 80 else ''}")
        print(f"    컨퍼런스: {paper.get('conference', 'N/A')}")
        print(f"    링크: {paper.get('link', 'N/A')}")
        
        intro = paper.get('introduction', '')
        if intro:
            intro_preview = intro.replace('\\n', ' ')[:150]
            print(f"    Introduction: {intro_preview}{'...' if len(intro) > 150 else ''}")
        
        qa_count = len(paper.get('QA', []))
        print(f"    Q&A 항목 수: {qa_count}")


def main():
    """메인 함수"""
    # 파일 경로 설정
    json_file = "paper_dataset/paperQA.json"
    
    print("🔍 JSON 파일 검증을 시작합니다...")
    print(f"파일: {json_file}")
    
    # JSON 파일 검증
    is_valid, message, papers = validate_json_file(json_file)
    
    if not is_valid:
        print(f"\n❌ {message}")
        sys.exit(1)
    
    print(f"\n✅ {message}")
    
    # 데이터 분석
    print("\n📊 데이터 분석 중...")
    analysis = analyze_paper_structure(papers)
    consistency_issues = check_data_consistency(papers)
    
    # 결과 출력
    print_analysis_report(analysis, consistency_issues)
    show_sample_papers(papers)
    
    # 추가 검증
    print(f"\n🔧 추가 검증:")
    total_chars = sum(len(paper.get('introduction', '')) for paper in papers)
    avg_intro_length = total_chars / len(papers) if papers else 0
    print(f"  • 평균 Introduction 길이: {avg_intro_length:.0f} 문자")
    
    # 특수 문자 검사
    special_chars = set()
    for paper in papers:
        intro = paper.get('introduction', '')
        for char in intro:
            if ord(char) > 127:  # ASCII가 아닌 문자
                special_chars.add(char)
    
    if special_chars:
        print(f"  • 발견된 특수 문자: {len(special_chars)}개")
        sample_chars = list(special_chars)[:10]
        print(f"    샘플: {', '.join(sample_chars)}")
    
    print(f"\n✨ 분석 완료!")


if __name__ == "__main__":
    main()
