"""Comprehensive tester for Biomedicina Postman Collection APIs.
Executes all endpoints from PubMed/PMC, ClinicalTrials.gov, and Europe PMC.
"""

import json
import httpx
from rich.console import Console
from rich.table import Table

console = Console()

NCBI_API_KEY = "1742f98e7b166a1aa284df7b00c9ae14ba09"
NCBI_EMAIL = "helixmind-research@example.com"
NCBI_TOOL = "helixmind-platform"
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

CTGOV_BASE = "https://clinicaltrials.gov/api/v2"

EPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
EPMC_ANNOTATIONS_BASE = "https://www.ebi.ac.uk/europepmc/annotations_api"

TERM = "breast cancer immunotherapy"

# Shared session variables
vars = {
    "pmid": "32117569",
    "pmc_uid": "7029759",
    "pmcid": "PMC7029759",
    "webenv": "",
    "query_key": "1",
    "nct_id": "NCT04280705",
    "ctgov_page_token": "",
    "epmc_source": "MED",
    "epmc_id": "32117569",
    "epmc_cursor": "*",
}

results = []

def test_endpoint(category, name, method, url, params=None, headers=None, extract_fn=None):
    console.print(f"[cyan]Testing [{category}] {name}...[/cyan]")
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            res = client.request(method, url, params=params, headers=headers)
            status = res.status_code
            content_type = res.headers.get("content-type", "")
            
            body_summary = ""
            if "json" in content_type:
                try:
                    data = res.json()
                    if extract_fn:
                        extract_fn(data)
                    # Create brief summary
                    keys = list(data.keys())[:6] if isinstance(data, dict) else f"list({len(data)})"
                    body_summary = f"JSON keys: {keys}"
                except Exception as e:
                    body_summary = f"JSON parse error: {e}"
            elif "xml" in content_type or "text" in content_type:
                snippet = res.text[:120].replace("\n", " ").strip()
                body_summary = f"Text/XML ({len(res.text)} chars): {snippet}..."
            elif "zip" in content_type:
                body_summary = f"ZIP binary stream ({len(res.content)} bytes)"
            else:
                body_summary = f"Bytes: {len(res.content)}, type: {content_type}"

            results.append({
                "category": category,
                "name": name,
                "status": status,
                "summary": body_summary,
                "success": 200 <= status < 300,
            })
            console.print(f"  [green]✓ {status}[/green] - {body_summary}")
    except Exception as e:
        results.append({
            "category": category,
            "name": name,
            "status": "ERROR",
            "summary": str(e),
            "success": False,
        })
        console.print(f"  [red]✗ Error: {e}[/red]")


def run_pubmed_tests():
    console.rule("[bold magenta]1. PubMed y PMC (NCBI E-utilities)[/bold magenta]")
    
    # 1. EInfo - listar DBs
    test_endpoint(
        "PubMed", "1. EInfo list databases", "GET", f"{NCBI_BASE}/einfo.fcgi",
        params={"retmode": "json", "api_key": NCBI_API_KEY, "tool": NCBI_TOOL, "email": NCBI_EMAIL}
    )

    # 2. EInfo - campos PubMed
    test_endpoint(
        "PubMed", "2. EInfo PubMed fields", "GET", f"{NCBI_BASE}/einfo.fcgi",
        params={"db": "pubmed", "retmode": "json", "api_key": NCBI_API_KEY, "tool": NCBI_TOOL, "email": NCBI_EMAIL}
    )

    # 3. ESearch PubMed - basic search
    def ext_search(d):
        r = d.get("esearchresult", {})
        if r.get("idlist"):
            vars["pmid"] = r["idlist"][0]
        if r.get("webenv"):
            vars["webenv"] = r["webenv"]
        if r.get("querykey"):
            vars["query_key"] = r["querykey"]
            
    test_endpoint(
        "PubMed", "3. ESearch basic", "GET", f"{NCBI_BASE}/esearch.fcgi",
        params={"db": "pubmed", "term": TERM, "retmode": "json", "retmax": "20", "retstart": "0", "api_key": NCBI_API_KEY, "tool": NCBI_TOOL, "email": NCBI_EMAIL},
        extract_fn=ext_search
    )

    # 4. ESearch PubMed - filtros fecha y tipo
    test_endpoint(
        "PubMed", "4. ESearch filtered", "GET", f"{NCBI_BASE}/esearch.fcgi",
        params={
            "db": "pubmed",
            "term": f"{TERM} AND (Review[Publication Type] OR Clinical Trial[Publication Type])",
            "retmode": "json", "retmax": "20", "sort": "pub_date", "datetype": "pdat",
            "mindate": "2024/01/01", "maxdate": "2026/12/31",
            "api_key": NCBI_API_KEY, "tool": NCBI_TOOL, "email": NCBI_EMAIL
        }
    )

    # 5. ESearch PubMed - History Server
    def ext_history(d):
        r = d.get("esearchresult", {})
        if r.get("webenv"):
            vars["webenv"] = r["webenv"]
        if r.get("querykey"):
            vars["query_key"] = r["querykey"]

    test_endpoint(
        "PubMed", "5. ESearch usehistory", "GET", f"{NCBI_BASE}/esearch.fcgi",
        params={"db": "pubmed", "term": TERM, "usehistory": "y", "retmode": "json", "retmax": "0", "api_key": NCBI_API_KEY, "tool": NCBI_TOOL, "email": NCBI_EMAIL},
        extract_fn=ext_history
    )

    # 6. ESummary - resumen estructurado
    test_endpoint(
        "PubMed", "6. ESummary", "GET", f"{NCBI_BASE}/esummary.fcgi",
        params={"db": "pubmed", "id": vars["pmid"], "retmode": "json", "version": "2.0", "api_key": NCBI_API_KEY, "tool": NCBI_TOOL, "email": NCBI_EMAIL}
    )

    # 7. EFetch PubMed - abstract texto
    test_endpoint(
        "PubMed", "7. EFetch abstract text", "GET", f"{NCBI_BASE}/efetch.fcgi",
        params={"db": "pubmed", "id": vars["pmid"], "rettype": "abstract", "retmode": "text", "api_key": NCBI_API_KEY, "tool": NCBI_TOOL, "email": NCBI_EMAIL}
    )

    # 8. EFetch PubMed - XML completo
    test_endpoint(
        "PubMed", "8. EFetch XML complete", "GET", f"{NCBI_BASE}/efetch.fcgi",
        params={"db": "pubmed", "id": vars["pmid"], "rettype": "xml", "retmode": "xml", "api_key": NCBI_API_KEY, "tool": NCBI_TOOL, "email": NCBI_EMAIL}
    )

    # 9. EFetch PubMed - lotes con WebEnv
    if vars["webenv"]:
        test_endpoint(
            "PubMed", "9. EFetch batch WebEnv", "GET", f"{NCBI_BASE}/efetch.fcgi",
            params={"db": "pubmed", "query_key": vars["query_key"], "WebEnv": vars["webenv"], "retstart": "0", "retmax": "5", "rettype": "abstract", "retmode": "text", "api_key": NCBI_API_KEY, "tool": NCBI_TOOL, "email": NCBI_EMAIL}
        )

    # 10. ESearch PMC
    def ext_pmc(d):
        r = d.get("esearchresult", {})
        if r.get("idlist"):
            vars["pmc_uid"] = r["idlist"][0]
            vars["pmcid"] = f"PMC{vars['pmc_uid']}"

    test_endpoint(
        "PubMed", "10. ESearch PMC open access", "GET", f"{NCBI_BASE}/esearch.fcgi",
        params={"db": "pmc", "term": f"{TERM} AND open access[filter]", "retmode": "json", "retmax": "10", "api_key": NCBI_API_KEY, "tool": NCBI_TOOL, "email": NCBI_EMAIL},
        extract_fn=ext_pmc
    )

    # 11. EFetch PMC - texto completo JATS XML
    test_endpoint(
        "PubMed", "11. EFetch PMC full JATS XML", "GET", f"{NCBI_BASE}/efetch.fcgi",
        params={"db": "pmc", "id": vars["pmc_uid"], "retmode": "xml", "api_key": NCBI_API_KEY, "tool": NCBI_TOOL, "email": NCBI_EMAIL}
    )

    # 12. ELink - relacionados neighbor
    test_endpoint(
        "PubMed", "12. ELink neighbor related", "GET", f"{NCBI_BASE}/elink.fcgi",
        params={"dbfrom": "pubmed", "db": "pubmed", "id": vars["pmid"], "cmd": "neighbor", "retmode": "json", "api_key": NCBI_API_KEY, "tool": NCBI_TOOL, "email": NCBI_EMAIL}
    )

    # 13. ELink - PubMed a PMC
    test_endpoint(
        "PubMed", "13. ELink PubMed to PMC", "GET", f"{NCBI_BASE}/elink.fcgi",
        params={"dbfrom": "pubmed", "db": "pmc", "id": vars["pmid"], "linkname": "pubmed_pmc", "retmode": "json", "api_key": NCBI_API_KEY, "tool": NCBI_TOOL, "email": NCBI_EMAIL}
    )

    # 14. ELink - citado por
    test_endpoint(
        "PubMed", "14. ELink cited in", "GET", f"{NCBI_BASE}/elink.fcgi",
        params={"dbfrom": "pubmed", "db": "pubmed", "id": vars["pmid"], "linkname": "pubmed_pubmed_citedin", "retmode": "json", "api_key": NCBI_API_KEY, "tool": NCBI_TOOL, "email": NCBI_EMAIL}
    )

    # 15. ESearch Cochrane
    test_endpoint(
        "PubMed", "15. ESearch Cochrane reviews", "GET", f"{NCBI_BASE}/esearch.fcgi",
        params={"db": "pubmed", "term": f'{TERM} AND "Cochrane Database Syst Rev"[Journal]', "retmode": "json", "retmax": "10", "sort": "pub_date", "api_key": NCBI_API_KEY, "tool": NCBI_TOOL, "email": NCBI_EMAIL}
    )


def run_clinicaltrials_tests():
    console.rule("[bold magenta]2. ClinicalTrials.gov (API v2)[/bold magenta]")

    # 1. Search studies by condition
    def ext_ct(d):
        if d.get("nextPageToken"):
            vars["ctgov_page_token"] = d["nextPageToken"]
        studies = d.get("studies", [])
        if studies:
            vars["nct_id"] = studies[0]["protocolSection"]["identificationModule"]["nctId"]

    test_endpoint(
        "ClinicalTrials", "1. Search by condition", "GET", f"{CTGOV_BASE}/studies",
        params={"query.cond": TERM, "pageSize": "10", "countTotal": "true", "format": "json"},
        extract_fn=ext_ct
    )

    # 2. Filter by recruitment status
    test_endpoint(
        "ClinicalTrials", "2. Filter overallStatus", "GET", f"{CTGOV_BASE}/studies",
        params={"query.cond": TERM, "filter.overallStatus": "RECRUITING,NOT_YET_RECRUITING", "pageSize": "10", "countTotal": "true"}
    )

    # 3. Filter by phase
    test_endpoint(
        "ClinicalTrials", "3. Filter phase advanced", "GET", f"{CTGOV_BASE}/studies",
        params={"query.cond": TERM, "filter.advanced": "AREA[Phase]PHASE3", "filter.overallStatus": "RECRUITING", "pageSize": "10", "countTotal": "true"}
    )

    # 4. By intervention and sorting
    test_endpoint(
        "ClinicalTrials", "4. By intervention and sort", "GET", f"{CTGOV_BASE}/studies",
        params={"query.cond": TERM, "query.intr": "pembrolizumab", "filter.overallStatus": "RECRUITING", "sort": "LastUpdatePostDate:desc", "pageSize": "10", "countTotal": "true"}
    )

    # 5. Field projection
    test_endpoint(
        "ClinicalTrials", "5. Fields projection", "GET", f"{CTGOV_BASE}/studies",
        params={"query.cond": TERM, "fields": "NCTId,BriefTitle,OverallStatus,Phase,LeadSponsorName,StartDate", "pageSize": "15"}
    )

    # 6. Pagination with pageToken
    if vars["ctgov_page_token"]:
        test_endpoint(
            "ClinicalTrials", "6. Pagination pageToken", "GET", f"{CTGOV_BASE}/studies",
            params={"query.cond": TERM, "pageSize": "10", "pageToken": vars["ctgov_page_token"]}
        )

    # 7. Study detail by NCT ID
    test_endpoint(
        "ClinicalTrials", "7. Study detail by NCT ID", "GET", f"{CTGOV_BASE}/studies/{vars['nct_id']}",
        params={"format": "json"}
    )

    # 8. Studies with results
    test_endpoint(
        "ClinicalTrials", "8. Studies with results", "GET", f"{CTGOV_BASE}/studies",
        params={"query.cond": TERM, "aggFilters": "results:with", "pageSize": "10", "countTotal": "true"}
    )

    # 9. Enums
    test_endpoint(
        "ClinicalTrials", "9. Enums", "GET", f"{CTGOV_BASE}/studies/enums"
    )

    # 10. Metadata
    test_endpoint(
        "ClinicalTrials", "10. Metadata", "GET", f"{CTGOV_BASE}/studies/metadata"
    )

    # 11. Stats size
    test_endpoint(
        "ClinicalTrials", "11. Stats size", "GET", f"{CTGOV_BASE}/stats/size"
    )

    # 12. Version
    test_endpoint(
        "ClinicalTrials", "12. API version", "GET", f"{CTGOV_BASE}/version"
    )


def run_europepmc_tests():
    console.rule("[bold magenta]3. Europe PMC (REST API)[/bold magenta]")

    # 1. Basic search (lite)
    def ext_epmc(d):
        if d.get("nextCursorMark"):
            vars["epmc_cursor"] = d["nextCursorMark"]
        results = d.get("resultList", {}).get("result", [])
        if results:
            vars["epmc_source"] = results[0].get("source", "MED")
            vars["epmc_id"] = results[0].get("id", vars["pmid"])
            if results[0].get("pmcid"):
                vars["pmcid"] = results[0]["pmcid"]

    test_endpoint(
        "EuropePMC", "1. Search lite", "GET", f"{EPMC_BASE}/search",
        params={"query": TERM, "format": "json", "resultType": "lite", "pageSize": "10"},
        extract_fn=ext_epmc
    )

    # 2. Core search (with full abstract)
    test_endpoint(
        "EuropePMC", "2. Search core open access", "GET", f"{EPMC_BASE}/search",
        params={"query": f"{TERM} AND OPEN_ACCESS:y", "format": "json", "resultType": "core", "pageSize": "5"}
    )

    # 3. Preprints only
    test_endpoint(
        "EuropePMC", "3. Preprints only (bioRxiv/medRxiv)", "GET", f"{EPMC_BASE}/search",
        params={"query": f'{TERM} AND SRC:"PPR"', "format": "json", "resultType": "lite", "pageSize": "10", "sort": "P_PDATE_D desc"}
    )

    # 4. PubMed open access + full text only
    test_endpoint(
        "EuropePMC", "4. PubMed open access + full text", "GET", f"{EPMC_BASE}/search",
        params={"query": f"{TERM} AND SRC:MED AND OPEN_ACCESS:y AND HAS_FT:y", "format": "json", "resultType": "lite", "pageSize": "10", "sort": "CITED desc"}
    )

    # 5. Pagination with cursorMark
    test_endpoint(
        "EuropePMC", "5. Pagination cursorMark", "GET", f"{EPMC_BASE}/search",
        params={"query": TERM, "format": "json", "resultType": "lite", "pageSize": "10", "cursorMark": vars["epmc_cursor"]}
    )

    # 6. Search by PMID
    test_endpoint(
        "EuropePMC", "6. By PMID EXT_ID", "GET", f"{EPMC_BASE}/search",
        params={"query": f"EXT_ID:{vars['pmid']} AND SRC:MED", "format": "json", "resultType": "core"}
    )

    # 7. Citations: articles citing this paper
    test_endpoint(
        "EuropePMC", "7. Citations", "GET", f"{EPMC_BASE}/{vars['epmc_source']}/{vars['epmc_id']}/citations",
        params={"format": "json", "page": "1", "pageSize": "10"}
    )

    # 8. References: articles cited by this paper
    test_endpoint(
        "EuropePMC", "8. References", "GET", f"{EPMC_BASE}/{vars['epmc_source']}/{vars['epmc_id']}/references",
        params={"format": "json", "page": "1", "pageSize": "10"}
    )

    # 9. Full-text JATS XML
    test_endpoint(
        "EuropePMC", "9. fullTextXML", "GET", f"{EPMC_BASE}/{vars['pmcid']}/fullTextXML"
    )

    # 10. Text-mined terms (textMinedTerms)
    test_endpoint(
        "EuropePMC", "10. textMinedTerms", "GET", f"{EPMC_BASE}/{vars['epmc_source']}/{vars['epmc_id']}/textMinedTerms",
        params={"format": "json"}
    )

    # 11. Database links
    test_endpoint(
        "EuropePMC", "11. databaseLinks", "GET", f"{EPMC_BASE}/{vars['epmc_source']}/{vars['epmc_id']}/databaseLinks",
        params={"format": "json", "page": "1", "pageSize": "10"}
    )

    # 12. Supplementary files
    test_endpoint(
        "EuropePMC", "12. Supplementary files", "HEAD", f"{EPMC_BASE}/{vars['pmcid']}/supplementaryFiles",
        params={"includeInlineImage": "false"}
    )

    # 13. Search fields
    test_endpoint(
        "EuropePMC", "13. Indexed fields", "GET", f"{EPMC_BASE}/fields",
        params={"format": "json"}
    )

    # 14. Annotations API
    test_endpoint(
        "EuropePMC", "14. Annotations API", "GET", f"{EPMC_ANNOTATIONS_BASE}/annotationsApi/annotationsByArticleIds",
        params={"articleIds": f"MED:{vars['pmid']}", "format": "JSON"}
    )


if __name__ == "__main__":
    console.print(f"[bold green]Starting comprehensive live API test with NCBI Key: {NCBI_API_KEY[:6]}...[/bold green]\n")
    run_pubmed_tests()
    run_clinicaltrials_tests()
    run_europepmc_tests()

    console.rule("[bold cyan]Test Summary[/bold cyan]")
    table = Table(title="Live Endpoint Execution Results", header_style="bold yellow")
    table.add_column("Category", style="cyan")
    table.add_column("Endpoint", style="white")
    table.add_column("Status", justify="center")
    table.add_column("Data Returned", style="dim")

    for r in results:
        status_style = "bold green" if r["success"] else "bold red"
        table.add_row(r["category"], r["name"], f"[{status_style}]{r['status']}[/{status_style}]", r["summary"][:60])

    console.print(table)

    # Also save detailed summary to JSON for deeper analysis
    with open("data/test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    console.print("[green]Saved results to data/test_results.json[/green]")
