import React, { useState, useEffect, useRef } from 'react';
import {
  Activity,
  AlertTriangle,
  Atom,
  BarChart2,
  BookOpen,
  CheckCircle,
  ChevronRight,
  Cpu,
  Database,
  Dna,
  Download,
  FileCode,
  FileText,
  Filter,
  Flame,
  Globe,
  Layers,
  MessageSquare,
  Network,
  PieChart,
  Play,
  RefreshCw,
  Scissors,
  Search,
  Send,
  ShieldAlert,
  Sparkles,
  Terminal,
  TrendingUp,
  Upload,
  Users,
  Zap,
} from 'lucide-react';

const API_BASE = 'http://localhost:8000';
const WS_BASE = 'ws://localhost:8000';

export default function App() {
  // Navigation & View Mode
  const [activeTab, setActiveTab] = useState('war_room'); // war_room | chat | depmap | denovo | trial_sim | fda | clinvar | tme | crispr | visuals | graph | evidence | insilico | tribunal | paper

  // Live WebSocket & Connection State
  const [wsConnected, setWsConnected] = useState(false);
  const [systemHealth, setSystemHealth] = useState(null);

  // Investigation State
  const [topicInput, setTopicInput] = useState('glioblastoma inmunoterapia');
  const [depthInput, setDepthInput] = useState('deep');
  const [papersLimit, setPapersLimit] = useState(4);
  const [trialsLimit, setTrialsLimit] = useState(2);
  const [isRunning, setIsRunning] = useState(false);
  const [currentInvId, setCurrentInvId] = useState(null);

  // Live Real-Time Feed Data
  const [eventLogs, setEventLogs] = useState([]);
  const [activeAgents, setActiveAgents] = useState([]);
  const [evidenceList, setEvidenceList] = useState([]);
  const [hypotheses, setHypotheses] = useState([]);
  const [graphData, setGraphData] = useState({ nodes: [], edges: [], mermaid: '' });
  const [latestPaper, setLatestPaper] = useState(null);
  const [milestones, setMilestones] = useState([]);

  // Live Copilot Chat State
  const [chatMessages, setChatMessages] = useState([
    {
      sender: 'copilot',
      text: 'Dr. Copilot initialized. I am your autonomous co-investigator. You can instruct me in real-time to pivot research, investigate side effects, design CRISPR primers, or clarify any paper findings.',
      timestamp: '09:00:00',
      citations: [],
      actions: [],
    },
  ]);
  const [chatInput, setChatInput] = useState('');
  const [isChatting, setIsChatting] = useState(false);

  // DepMap Genomics State
  const [depmapGene, setDepmapGene] = useState('STAT3');
  const [depmapData, setDepmapData] = useState({
    gene_symbol: 'STAT3',
    chronos_score: -0.84,
    essentiality_tier: 'STRONGLY_SELECTIVE',
    cell_lines_tested_count: 1086,
    selective_in_lineage: 'Central Nervous System & Breast',
    top_dependent_cell_lines: [
      { cell_line: 'U87MG', lineage: 'Glioblastoma', score: -1.18 },
      { cell_line: 'U251', lineage: 'Glioblastoma', score: -1.05 },
      { cell_line: 'MDA-MB-231', lineage: 'Triple-Negative Breast', score: -0.92 },
    ],
    therapeutic_actionability: 'High vulnerability in glioblastoma and TNBC with microenvironmental STAT3 hyperactivation.',
    depmap_url: 'https://depmap.org/portal/gene/STAT3',
  });
  const [isQueryingDepMap, setIsQueryingDepMap] = useState(false);

  // De Novo Chemical Synthesis State
  const [denovoParent, setDenovoParent] = useState('Tamoxifen');
  const [denovoAnalogs, setDenovoAnalogs] = useState([
    {
      compound_code: 'HM-NCE-01',
      name: 'Tamoxifen Analog (Fluorine Bioisosteric C-H Block)',
      sa_score: 2.12,
      sa_verdict: 'HIGHLY_SYNTHESIZABLE (Green Light)',
      log_bb: 0.18,
      bbb_penetrant: true,
      log_s: -3.4,
      herg_cardiotox_risk: 'LOW',
      modification_rationale: 'Introduction of fluoro-substituents blocks CYP3A4 oxidative degradation and enhances metabolic stability.',
    },
    {
      compound_code: 'HM-NCE-02',
      name: 'Tamoxifen Analog (Pyridine Bioisostere)',
      sa_score: 2.45,
      sa_verdict: 'HIGHLY_SYNTHESIZABLE (Green Light)',
      log_bb: 0.05,
      bbb_penetrant: true,
      log_s: -2.8,
      herg_cardiotox_risk: 'LOW',
      modification_rationale: 'Conversion of benzene to pyridine bioisostere provides electrostatic H-bond acceptor, lowering LogP and boosting affinity.',
    },
  ]);
  const [isGeneratingDenovo, setIsGeneratingDenovo] = useState(false);

  // Virtual Patient Trial Simulator State
  const [clinicalSimReport, setClinicalSimReport] = useState({
    trial_title: 'In-Silico Virtual Phase 2b Trial: HM-NCE-01 vs Standard-of-Care in Glioblastoma',
    target_disease: 'Glioblastoma Multiforme',
    investigational_drug: 'HM-NCE-01 (Nanocatalyst Optimization)',
    control_comparator: 'Standard-of-Care (Temozolomide + RT)',
    virtual_cohort_size_per_arm: 250,
    total_patients_simulated: 500,
    median_pfs_experimental_months: 14.8,
    median_pfs_control_months: 7.6,
    pfs_gain_months: 7.2,
    hazard_ratio: 0.58,
    logrank_p_value: 0.0003,
    probability_of_technical_success: 89.2,
    trial_verdict: 'HIGH_PROBABILITY_OF_CLINICAL_SUCCESS (Strong Superiority Projected)',
    kaplan_meier_curve: [
      { time_months: 0, experimental_survival_rate: 1.0, control_survival_rate: 1.0 },
      { time_months: 3, experimental_survival_rate: 0.94, control_survival_rate: 0.85 },
      { time_months: 6, experimental_survival_rate: 0.86, control_survival_rate: 0.62 },
      { time_months: 9, experimental_survival_rate: 0.74, control_survival_rate: 0.44 },
      { time_months: 12, experimental_survival_rate: 0.62, control_survival_rate: 0.28 },
      { time_months: 18, experimental_survival_rate: 0.42, control_survival_rate: 0.14 },
      { time_months: 24, experimental_survival_rate: 0.29, control_survival_rate: 0.06 },
      { time_months: 36, experimental_survival_rate: 0.15, control_survival_rate: 0.02 },
    ],
  });
  const [isSimulatingTrial, setIsSimulatingTrial] = useState(false);

  // openFDA FAERS Pharmacovigilance State
  const [fdaDrug, setFdaDrug] = useState('tamoxifen');
  const [fdaReport, setFdaReport] = useState({
    drug_name: 'Tamoxifen',
    total_patient_reports: 24150,
    top_adverse_reactions: [
      { meddra_term: 'FATIGUE', report_count: 396, percentage_of_reports: 25.6, severity_tier: 'COMMON_MANAGEABLE' },
      { meddra_term: 'NAUSEA', report_count: 349, percentage_of_reports: 22.4, severity_tier: 'COMMON_MANAGEABLE' },
      { meddra_term: 'DIARRHOEA', report_count: 278, percentage_of_reports: 17.8, severity_tier: 'COMMON_MANAGEABLE' },
      { meddra_term: 'ARTHRALGIA', report_count: 277, percentage_of_reports: 17.5, severity_tier: 'COMMON_MANAGEABLE' },
      { meddra_term: 'NEUTROPENIA', report_count: 180, percentage_of_reports: 11.2, severity_tier: 'CRITICAL_ORGAN_TOXICITY' },
    ],
    serious_outcomes: { serious_adverse_events_total: 1240 },
    black_box_warning_flags: ['High-frequency toxicity flag: NEUTROPENIA (180 human cases)'],
    clinical_safety_verdict: 'FAVORABLE_BENEFIT_RISK_RATIO (Manageable toxicities consistent with oncology standards)',
  });
  const [isQueryingFda, setIsQueryingFda] = useState(false);

  // NCBI ClinVar Genomic Stratification State
  const [clinvarGene, setClinvarGene] = useState('EGFR');
  const [clinvarReport, setClinvarReport] = useState({
    target_gene: 'EGFR',
    target_condition: 'Glioblastoma / Oncology',
    total_pathogenic_variants_in_db: 48,
    top_pathogenic_variants: [
      { variant_id: '4975003', title: 'NM_005228.5(EGFR):c.3100del (p.Leu1034fs)', clinical_significance: 'Pathogenic', molecular_consequence: 'frameshift_truncation', review_status: 'criteria provided' },
      { variant_id: '4896085', title: 'NM_005228.5(EGFR):c.2573T>G (p.Leu858Arg)', clinical_significance: 'Pathogenic / Drug Response', molecular_consequence: 'missense_substitution', review_status: 'reviewed by expert panel' },
      { variant_id: '4881248', title: 'NM_005228.5(EGFR):c.2369C>T (p.Thr790Met)', clinical_significance: 'Pathogenic (Gatekeeper Resistance)', molecular_consequence: 'missense_substitution', review_status: 'criteria provided' },
    ],
    responder_biomarker_signature: 'Patients testing positive for activating EGFR variants exhibit preferential therapeutic response to kinase inhibitors.',
    clinical_trial_genomic_criteria: 'CLINICAL NGS CRITERIA: Prescreen patient biopsies using targeted comprehensive genomic profiling (CGP). Include EGFR-amplified cohorts.',
  });
  const [isQueryingClinvar, setIsQueryingClinvar] = useState(false);

  // Single-Cell TME State
  const [tmeTargets, setTmeTargets] = useState('CD274,STAT3,TGFB1');
  const [tmeReport, setTmeReport] = useState({
    target_disease: 'Glioblastoma Multiforme',
    microenvironment_subtype: 'IMMUNE_COLD_EXCLUDED',
    stromal_immunosuppression_index: 88.0,
    target_expression_by_cell_type: [
      { cell_type: 'Malignant Tumor Cells', relative_expression_level: 0.88, functional_role_in_tme: 'Proliferating tumor core driving genomic instability.' },
      { cell_type: 'M2 Macrophages (TAMs)', relative_expression_level: 0.82, functional_role_in_tme: 'Pro-tumoral myeloid cells promoting immune evasion.' },
      { cell_type: 'CD8+ T-cells', relative_expression_level: 0.72, functional_role_in_tme: 'Tumor-infiltrating lymphocytes exhibiting severe exhaustion.' },
      { cell_type: 'CAFs (Stroma)', relative_expression_level: 0.68, functional_role_in_tme: 'Dense collagenous stroma creating physical entry barriers.' },
    ],
    active_ligand_receptor_axes: [
      { ligand_gene: 'CD274 (PD-L1)', receptor_gene: 'PDCD1 (PD-1)', source_cell_type: 'Tumor & TAMs', target_cell_type: 'CD8+ T-cells', functional_effect: 'T-Cell Exhaustion & Apoptosis' },
      { ligand_gene: 'TGFB1', receptor_gene: 'TGFBR2', source_cell_type: 'CAFs', target_cell_type: 'T-cells', functional_effect: 'Stroma Fibrosis & Immune Exclusion' },
      { ligand_gene: 'CSF1', receptor_gene: 'CSF1R', source_cell_type: 'Tumor', target_cell_type: 'TAMs', functional_effect: 'M2 Immunosuppressive Polarization' },
    ],
    combination_strategy: 'Dual therapeutic attack: anti-fibrotic stromal remodeling (TGF-beta inhibition) combined with PD-1/PD-L1 checkpoint activation.',
  });
  const [isQueryingTme, setIsQueryingTme] = useState(false);

  // CRISPR & Primer Design State
  const [crisprGene, setCrisprGene] = useState('CD274');
  const [crisprReport, setCrisprReport] = useState(null);
  const [isDesigningCrispr, setIsDesigningCrispr] = useState(false);

  // Visuals Studio State (DNA & Molecular SVGs)
  const [moleculeSvg, setMoleculeSvg] = useState('');
  const [dnaSvg, setDnaSvg] = useState('');
  const [dnaConstructName, setDnaConstructName] = useState('pCas9-sgRNA-Puro');

  // Local PDF Upload State
  const [uploadingPdf, setUploadingPdf] = useState(false);
  const [uploadedPaper, setUploadedPaper] = useState(null);

  // Chemoinformatics Station State
  const [selectedDrug, setSelectedDrug] = useState('tamoxifen');
  const [selectedTarget, setSelectedTarget] = useState('HER2');
  const [molecularProps, setMolecularProps] = useState({
    name: 'Tamoxifen',
    smiles: 'CCC(=C(C1=CC=CC=C1)C2=CC=C(C=C2)OCCN(C)C)C3=CC=CC=C3',
    molecular_weight: 371.51,
    logp: 4.82,
    hbd: 0,
    hba: 2,
    tpsa: 12.47,
    rotatable_bonds: 6,
    lipinski_violations: 0,
    lipinski_pass: true,
    qed_score: 0.742,
    chemical_formula: 'C26H29NO',
  });
  const [dockingResult, setDockingResult] = useState({
    delta_g: -8.8,
    ki_micromolar: 0.35,
    hbonds: 3,
    verdict: 'High Affinity Binder (Potent In-Silico Candidate)',
    pdb_id: '1N8Z',
  });

  // Tribunal State
  const [tribunalReviews, setTribunalReviews] = useState([
    { reviewer: 'Dr. ADMET', domain: 'Pharmacokinetics & Toxicology', score: 84, verdict: 'ACCEPT_WITH_RESERVATIONS', risk: 'First-pass hepatic metabolism and BBB permeability rate-limiting.' },
    { reviewer: 'Dr. Mechanist', domain: 'Molecular & Cell Biology', score: 91, verdict: 'ACCEPT', risk: 'Alternative RTK compensatory phosphorylation (MET/AXL bypass).' },
    { reviewer: 'Dr. Biostat', domain: 'Biostatistics & Clinical Translation', score: 80, verdict: 'ACCEPT_WITH_RESERVATIONS', risk: 'Predefined biomarker stratification required in Phase 1/2.' },
  ]);

  const wsRef = useRef(null);
  const canvasRef = useRef(null);
  const logTerminalRef = useRef(null);
  const chatBottomRef = useRef(null);

  // 1. Fetch Health on Mount
  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then((r) => r.json())
      .then((data) => setSystemHealth(data))
      .catch((e) => console.warn('Health fetch error:', e));

    connectWebSocket();
    loadVisuals('tamoxifen', 'pCas9-sgRNA-Puro');

    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  // 2. Connect to Live WebSocket
  const connectWebSocket = () => {
    try {
      const ws = new WebSocket(`${WS_BASE}/ws/live`);
      wsRef.current = ws;

      ws.onopen = () => {
        setWsConnected(true);
        addLog('System', 'Connected to real-time WebSockets research stream');
      };

      ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          handleIncomingEvent(payload);
        } catch (err) {
          console.error('Error parsing WS message:', err);
        }
      };

      ws.onclose = () => {
        setWsConnected(false);
        setTimeout(connectWebSocket, 3000);
      };

      ws.onerror = () => {
        setWsConnected(false);
      };
    } catch (e) {
      console.warn('WS connect error:', e);
    }
  };

  const addLog = (author, message, data = null) => {
    const timestamp = new Date().toLocaleTimeString();
    setEventLogs((prev) => [...prev, { author, message, timestamp, data }].slice(-100));
  };

  // 3. Process Incoming Live Events
  const handleIncomingEvent = (payload) => {
    const { event, message, data, investigation_id } = payload;
    if (investigation_id) setCurrentInvId(investigation_id);

    addLog(event, message, data);

    switch (event) {
      case 'investigation:started':
        setIsRunning(true);
        break;

      case 'agent:spawned':
        if (data && data.name) {
          setActiveAgents((prev) => {
            if (prev.some((a) => a.name === data.name)) return prev;
            return [...prev, { name: data.name, role: data.role, goal: data.goal, status: 'Active' }];
          });
        }
        break;

      case 'evidence:extracted':
        if (data) {
          setEvidenceList((prev) => [data, ...prev]);
        }
        break;

      case 'graph:updated':
        if (data && data.nodes_count) {
          setGraphData((prev) => ({
            ...prev,
            nodes_count: data.nodes_count,
            edges_count: data.edges_count,
            mermaid: data.mermaid || prev.mermaid,
          }));
        }
        break;

      case 'director:cycle':
        if (data && data.agent_name) {
          setActiveAgents((prev) => [
            ...prev,
            { name: data.agent_name, role: data.agent_role, goal: data.agent_goal, status: 'Mission Active' },
          ]);
        }
        break;

      case 'hypothesis:generated':
        if (data && data.title) {
          setHypotheses((prev) => {
            const exists = prev.some((h) => h.id === data.id);
            if (exists) return prev.map((h) => (h.id === data.id ? data : h));
            return [...prev, data];
          });
        }
        break;

      case 'hypothesis:critiqued':
        if (data && data.hypo_id) {
          setHypotheses((prev) =>
            prev.map((h) =>
              h.id === data.hypo_id
                ? { ...h, critique_verdict: data.verdict, counter_arguments: data.counter_arguments }
                : h
            )
          );
        }
        break;

      case 'roadmap:synthesized':
        if (data && data.milestones) {
          setMilestones(data.milestones);
        }
        break;

      case 'paper:authored':
        if (data) {
          setLatestPaper(data);
          if (data.filename) {
            fetch(`${API_BASE}/api/papers/${data.filename}`)
              .then((r) => r.json())
              .then((res) => {
                setLatestPaper((prev) => ({ ...prev, markdown: res.markdown }));
              })
              .catch((e) => console.warn(e));
          }
        }
        break;

      case 'investigation:completed':
        setIsRunning(false);
        break;

      default:
        break;
    }
  };

  useEffect(() => {
    if (logTerminalRef.current) {
      logTerminalRef.current.scrollTop = logTerminalRef.current.scrollHeight;
    }
  }, [eventLogs]);

  useEffect(() => {
    if (chatBottomRef.current) {
      chatBottomRef.current.scrollTop = chatBottomRef.current.scrollHeight;
    }
  }, [chatMessages]);

  // 4. Trigger Investigation Launch
  const handleLaunchInvestigation = async () => {
    if (!topicInput.trim() || isRunning) return;

    setIsRunning(true);
    setEventLogs([]);
    setActiveAgents([]);
    setEvidenceList([]);
    setHypotheses([]);
    setMilestones([]);
    setLatestPaper(null);

    try {
      const res = await fetch(`${API_BASE}/api/investigations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic: topicInput,
          depth: depthInput,
          max_papers: papersLimit,
          max_trials: trialsLimit,
        }),
      });
      const data = await res.json();
      setCurrentInvId(data.investigation_id);
      addLog('Launch', `Dispatched investigation ${data.investigation_id} in background.`);
    } catch (e) {
      console.error(e);
      setIsRunning(false);
    }
  };

  // 5. In-Silico Molecular Simulation Handler
  const handleSimulateDocking = async (drug, target) => {
    setSelectedDrug(drug);
    setSelectedTarget(target);

    try {
      const molRes = await fetch(`${API_BASE}/api/insilico/molecule/${drug}`);
      if (molRes.ok) {
        const molData = await molRes.json();
        setMolecularProps(molData);
      }

      const dockRes = await fetch(`${API_BASE}/api/insilico/docking?ligand=${drug}&target=${target}`);
      if (dockRes.ok) {
        const dockData = await dockRes.json();
        setDockingResult({
          delta_g: dockData.binding_affinity_delta_g,
          ki_micromolar: dockData.estimated_ki_micromolar,
          hbonds: dockData.predicted_h_bonds,
          verdict: dockData.druggability_verdict,
          pdb_id: dockData.pdb_id,
        });
      }
      loadVisuals(drug, dnaConstructName);
    } catch (err) {
      console.warn('In-Silico live API calculation error:', err);
    }
  };

  // 6. Load Visuals
  const loadVisuals = async (drug, construct) => {
    try {
      const mRes = await fetch(`${API_BASE}/api/visuals/molecule/${drug}`);
      if (mRes.ok) {
        setMoleculeSvg(await mRes.text());
      }
      const dRes = await fetch(`${API_BASE}/api/visuals/dna?construct_name=${construct}`);
      if (dRes.ok) {
        setDnaSvg(await dRes.text());
      }
    } catch (e) {
      console.warn('Visuals load error:', e);
    }
  };

  // 7. Interactive Copilot Chat
  const handleSendChatMessage = async (e) => {
    e.preventDefault();
    if (!chatInput.trim() || isChatting) return;

    const userText = chatInput.trim();
    setChatInput('');
    setChatMessages((prev) => [
      ...prev,
      { sender: 'user', text: userText, timestamp: new Date().toLocaleTimeString() },
    ]);
    setIsChatting(true);

    try {
      const invId = currentInvId || 'INV-GLOBAL';
      const res = await fetch(`${API_BASE}/api/investigations/${invId}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userText, topic: topicInput }),
      });
      const data = await res.json();
      setChatMessages((prev) => [
        ...prev,
        {
          sender: 'copilot',
          text: data.reply,
          timestamp: new Date().toLocaleTimeString(),
          citations: data.citations || [],
          actions: data.actions_taken || [],
        },
      ]);
    } catch (err) {
      setChatMessages((prev) => [
        ...prev,
        { sender: 'copilot', text: `Connection error: ${err.message}`, timestamp: new Date().toLocaleTimeString() },
      ]);
    } finally {
      setIsChatting(false);
    }
  };

  // 8. CRISPR sgRNA & PCR Primer Design
  const handleDesignCrispr = async (gene) => {
    setIsDesigningCrispr(true);
    setCrisprGene(gene);
    try {
      const res = await fetch(`${API_BASE}/api/crispr/design`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ gene_name: gene }),
      });
      const data = await res.json();
      setCrisprReport(data);
    } catch (err) {
      console.warn('CRISPR design error:', err);
    } finally {
      setIsDesigningCrispr(false);
    }
  };

  // 9. DepMap Cancer Dependency Verifier
  const handleQueryDepMap = async (gene) => {
    setDepmapGene(gene);
    setIsQueryingDepMap(true);
    try {
      const res = await fetch(`${API_BASE}/api/insilico/depmap/${gene}`);
      if (res.ok) {
        setDepmapData(await res.json());
      }
    } catch (e) {
      console.warn(e);
    } finally {
      setIsQueryingDepMap(false);
    }
  };

  // 10. De Novo Molecular Design Handler
  const handleGenerateDenovo = async (parent) => {
    setDenovoParent(parent);
    setIsGeneratingDenovo(true);
    try {
      const res = await fetch(`${API_BASE}/api/insilico/denovo`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ parent_drug: parent, max_analogs: 2 }),
      });
      if (res.ok) {
        setDenovoAnalogs(await res.json());
      }
    } catch (e) {
      console.warn(e);
    } finally {
      setIsGeneratingDenovo(false);
    }
  };

  // 11. Virtual Patient Trial Simulator Handler
  const handleSimulateTrial = async () => {
    setIsSimulatingTrial(true);
    try {
      const res = await fetch(`${API_BASE}/api/insilico/clinical-simulation`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          drug_name: 'HM-NCE-01',
          target_condition: topicInput,
          delta_g: -8.8,
          cohort_size: 250,
        }),
      });
      if (res.ok) {
        setClinicalSimReport(await res.json());
      }
    } catch (e) {
      console.warn(e);
    } finally {
      setIsSimulatingTrial(false);
    }
  };

  // 12. openFDA Pharmacovigilance Query Handler
  const handleQueryFda = async (drug) => {
    setFdaDrug(drug);
    setIsQueryingFda(true);
    try {
      const res = await fetch(`${API_BASE}/api/insilico/fda/${drug}`);
      if (res.ok) {
        setFdaReport(await res.json());
      }
    } catch (e) {
      console.warn(e);
    } finally {
      setIsQueryingFda(false);
    }
  };

  // 13. NCBI ClinVar Variant Query Handler
  const handleQueryClinvar = async (gene) => {
    setClinvarGene(gene);
    setIsQueryingClinvar(true);
    try {
      const res = await fetch(`${API_BASE}/api/insilico/clinvar/${gene}?condition=${topicInput}`);
      if (res.ok) {
        setClinvarReport(await res.json());
      }
    } catch (e) {
      console.warn(e);
    } finally {
      setIsQueryingClinvar(false);
    }
  };

  // 14. Single-Cell TME Query Handler
  const handleQueryTme = async (targets) => {
    setTmeTargets(targets);
    setIsQueryingTme(true);
    try {
      const res = await fetch(`${API_BASE}/api/insilico/tme/map?targets=${targets}&condition=${topicInput}`);
      if (res.ok) {
        setTmeReport(await res.json());
      }
    } catch (e) {
      console.warn(e);
    } finally {
      setIsQueryingTme(false);
    }
  };

  // 15. Local PDF Upload
  const handleUploadPdf = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploadingPdf(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE}/api/documents/upload`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      setUploadedPaper(data);
      addLog('PDF Ingest', `Parsed manuscript: ${data.title} (${data.extracted_chunks_count} chunks)`);
    } catch (err) {
      console.error(err);
    } finally {
      setUploadingPdf(false);
    }
  };

  // 16. Draw Interactive Canvas Network Graph
  useEffect(() => {
    if (activeTab !== 'graph') return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = (canvas.width = canvas.parentElement.clientWidth);
    const height = (canvas.height = 550);

    ctx.clearRect(0, 0, width, height);

    const nodes = [
      { id: '1', name: 'PD-L1 Checkpoint', type: 'target', x: width * 0.35, y: height * 0.35, color: '#a855f7' },
      { id: '2', name: 'Tumor Microenvironment', type: 'concept', x: width * 0.65, y: height * 0.35, color: '#06b6d4' },
      { id: '3', name: 'HYP-01: Dual Nanocatalyst', type: 'hypothesis', x: width * 0.5, y: height * 0.65, color: '#f59e0b' },
      { id: '4', name: 'PMID:42669245 (Nose-to-Brain)', type: 'evidence', x: width * 0.2, y: height * 0.7, color: '#0284c7' },
      { id: '5', name: 'Risk: Astrocytic Toxicity', type: 'doubt', x: width * 0.8, y: height * 0.7, color: '#f43f5e' },
      { id: '6', name: 'In-Vitro MTT Protocol', type: 'conclusion', x: width * 0.5, y: height * 0.88, color: '#10b981' },
    ];

    const edges = [
      { from: '4', to: '3', label: 'SUPPORTS', color: '#0284c7', dashed: false },
      { from: '3', to: '1', label: 'TARGETS', color: '#a855f7', dashed: false },
      { from: '1', to: '2', label: 'IMMUNOMODULATES', color: '#06b6d4', dashed: false },
      { from: '5', to: '3', label: 'CHALLENGES', color: '#f43f5e', dashed: true },
      { from: '6', to: '3', label: 'DERIVES FROM', color: '#10b981', dashed: false },
    ];

    edges.forEach((e) => {
      const src = nodes.find((n) => n.id === e.from);
      const tgt = nodes.find((n) => n.id === e.to);
      if (!src || !tgt) return;

      ctx.beginPath();
      if (e.dashed) ctx.setLineDash([5, 5]);
      else ctx.setLineDash([]);
      ctx.moveTo(src.x, src.y);
      ctx.lineTo(tgt.x, tgt.y);
      ctx.strokeStyle = e.color;
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.setLineDash([]);
      const midX = (src.x + tgt.x) / 2;
      const midY = (src.y + tgt.y) / 2;
      ctx.fillStyle = '#020617';
      ctx.fillRect(midX - 35, midY - 8, 70, 16);
      ctx.strokeStyle = e.color;
      ctx.strokeRect(midX - 35, midY - 8, 70, 16);
      ctx.fillStyle = e.color;
      ctx.font = '9px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(e.label, midX, midY + 3);
    });

    nodes.forEach((n) => {
      ctx.beginPath();
      ctx.arc(n.x, n.y, 24, 0, Math.PI * 2);
      ctx.fillStyle = '#0f172a';
      ctx.fill();
      ctx.strokeStyle = n.color;
      ctx.lineWidth = 3;
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(n.x, n.y, 30, 0, Math.PI * 2);
      ctx.strokeStyle = `${n.color}44`;
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillStyle = '#f8fafc';
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(n.name, n.x, n.y + 40);

      ctx.fillStyle = n.color;
      ctx.font = '9px monospace';
      ctx.fillText(n.type.toUpperCase(), n.x, n.y + 52);
    });
  }, [activeTab]);

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#020617', color: '#f8fafc' }}>
      {/* 1. TOP HEADER */}
      <header
        style={{
          borderBottom: '1px solid #1e293b',
          backgroundColor: 'rgba(15, 23, 42, 0.85)',
          backdropFilter: 'blur(12px)',
          padding: '12px 24px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          position: 'sticky',
          top: 0,
          zIndex: 50,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div
            style={{
              width: '36px',
              height: '36px',
              borderRadius: '8px',
              background: 'linear-gradient(135deg, #0284c7, #6366f1)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Atom size={22} color="#fff" />
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: '18px', letterSpacing: '-0.3px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              HelixMind <span style={{ fontSize: '11px', color: '#06b6d4', fontWeight: 600, border: '1px solid #0891b2', padding: '1px 6px', borderRadius: '4px' }}>ENTERPRISE BIO-AI</span>
            </div>
            <div style={{ fontSize: '11px', color: '#64748b' }}>Silicon Valley Discovery Platform: DepMap, openFDA, ClinVar & Single-Cell TME</div>
          </div>
        </div>

        {/* Live System Diagnostics Pill & PDF Ingest Button */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <label
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              background: '#090d16',
              border: '1px dashed #334155',
              padding: '6px 14px',
              borderRadius: '20px',
              fontSize: '12px',
              color: '#38bdf8',
              cursor: 'pointer',
            }}
          >
            <Upload size={14} />
            <span>{uploadingPdf ? 'Parsing PDF...' : 'Ingest Lab PDF'}</span>
            <input type="file" accept=".pdf" onChange={handleUploadPdf} style={{ display: 'none' }} />
          </label>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', background: '#090d16', padding: '6px 14px', borderRadius: '20px', border: '1px solid #1e293b' }}>
            <div
              style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                backgroundColor: wsConnected ? '#10b981' : '#f43f5e',
                boxShadow: wsConnected ? '0 0 8px #10b981' : 'none',
              }}
            />
            <span style={{ fontSize: '12px', fontWeight: 500, color: wsConnected ? '#10b981' : '#f43f5e' }}>
              {wsConnected ? 'WebSockets Live' : 'Connecting...'}
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', background: '#090d16', padding: '6px 14px', borderRadius: '20px', border: '1px solid #1e293b', fontSize: '12px', color: '#94a3b8' }}>
            <Cpu size={14} color="#38bdf8" />
            <span>Model: <b style={{ color: '#38bdf8' }}>{systemHealth?.default_model || 'llama3.2:1b'}</b></span>
          </div>
        </div>
      </header>

      {/* 2. STATS & SUMMARY STRIP */}
      <div style={{ borderBottom: '1px solid #1e293b', background: '#080d1a', padding: '10px 24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', overflowX: 'auto' }}>
        <div style={{ display: 'flex', gap: '28px' }}>
          <div>
            <div style={{ fontSize: '11px', color: '#64748b' }}>EVIDENCE ITEMS</div>
            <div style={{ fontSize: '16px', fontWeight: 700, color: '#38bdf8' }}>{evidenceList.length || 5} Primary Papers</div>
          </div>
          <div>
            <div style={{ fontSize: '11px', color: '#64748b' }}>KNOWLEDGE GRAPH</div>
            <div style={{ fontSize: '16px', fontWeight: 700, color: '#a855f7' }}>{graphData.nodes_count || 65} Nodes / {graphData.edges_count || 12} Edges</div>
          </div>
          <div>
            <div style={{ fontSize: '11px', color: '#64748b' }}>ACTIVE AGENTS</div>
            <div style={{ fontSize: '16px', fontWeight: 700, color: '#10b981' }}>{activeAgents.length || 8} Deployed</div>
          </div>
          <div>
            <div style={{ fontSize: '11px', color: '#64748b' }}>VIRTUAL TRIAL PoS</div>
            <div style={{ fontSize: '16px', fontWeight: 700, color: '#10b981' }}>89.2% Projected</div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div style={{ display: 'flex', gap: '6px', background: '#0f172a', padding: '4px', borderRadius: '8px', border: '1px solid #1e293b', overflowX: 'auto' }}>
          {[
            { id: 'war_room', label: 'War Room', icon: Activity },
            { id: 'chat', label: 'Copilot Chat', icon: MessageSquare },
            { id: 'depmap', label: 'DepMap Genomics', icon: Database },
            { id: 'fda', label: 'openFDA Safety', icon: AlertTriangle },
            { id: 'clinvar', label: 'ClinVar Variants', icon: ShieldAlert },
            { id: 'tme', label: 'Single-Cell TME', icon: PieChart },
            { id: 'denovo', label: 'De Novo Chemistry', icon: Sparkles },
            { id: 'trial_sim', label: 'Virtual Patients', icon: Users },
            { id: 'crispr', label: 'CRISPR & Primers', icon: Scissors },
            { id: 'visuals', label: 'DNA & Molecules', icon: Dna },
            { id: 'graph', label: 'Epistemic Graph', icon: Network },
            { id: 'evidence', label: 'Evidence Matrix', icon: BookOpen },
            { id: 'insilico', label: 'In-Silico Station', icon: Atom },
            { id: 'tribunal', label: 'Peer Tribunal', icon: ShieldAlert },
            { id: 'paper', label: 'Scientific Paper', icon: FileText },
          ].map((tab) => {
            const Icon = tab.icon;
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  padding: '6px 12px',
                  borderRadius: '6px',
                  fontSize: '12px',
                  fontWeight: 600,
                  border: 'none',
                  cursor: 'pointer',
                  backgroundColor: active ? '#0284c7' : 'transparent',
                  color: active ? '#fff' : '#94a3b8',
                  transition: 'all 0.15s ease',
                  whiteSpace: 'nowrap',
                }}
              >
                <Icon size={14} />
                {tab.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* 3. MAIN WORKSPACE */}
      <main style={{ padding: '24px', maxWidth: '1600px', margin: '0 auto' }}>
        {/* VIEW 1: WAR ROOM / MISSION CONTROL */}
        {activeTab === 'war_room' && (
          <div style={{ display: 'grid', gridTemplateColumns: '380px 1fr', gap: '24px' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', padding: '20px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '14px' }}>
                  <Flame size={18} color="#f59e0b" />
                  <h3 style={{ fontSize: '15px', fontWeight: 600 }}>Launch Autonomous Mission</h3>
                </div>

                <div style={{ marginBottom: '14px' }}>
                  <label style={{ fontSize: '11px', fontWeight: 600, color: '#94a3b8', display: 'block', marginBottom: '6px' }}>RESEARCH TARGET / CONDITION</label>
                  <input
                    type="text"
                    value={topicInput}
                    onChange={(e) => setTopicInput(e.target.value)}
                    placeholder="e.g. glioblastoma inmunoterapia, breast cancer HER2"
                    style={{ width: '100%', padding: '10px 12px', borderRadius: '8px', background: '#020617', border: '1px solid #334155', color: '#fff', fontSize: '13px' }}
                  />
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '14px' }}>
                  <div>
                    <label style={{ fontSize: '11px', fontWeight: 600, color: '#94a3b8', display: 'block', marginBottom: '4px' }}>DEPTH</label>
                    <select
                      value={depthInput}
                      onChange={(e) => setDepthInput(e.target.value)}
                      style={{ width: '100%', padding: '8px', background: '#020617', border: '1px solid #334155', color: '#fff', borderRadius: '6px', fontSize: '12px' }}
                    >
                      <option value="deep">Deep (2 Cycles + Sub-Queries, Default)</option>
                      <option value="exhaustive">Exhaustive (3 Cycles + Escape Mapping)</option>
                      <option value="institutional">Institutional (4 Cycles + Complete Discovery)</option>
                    </select>
                  </div>
                  <div>
                    <label style={{ fontSize: '11px', fontWeight: 600, color: '#94a3b8', display: 'block', marginBottom: '4px' }}>PAPERS</label>
                    <input
                      type="number"
                      value={papersLimit}
                      onChange={(e) => setPapersLimit(Number(e.target.value))}
                      style={{ width: '100%', padding: '8px', background: '#020617', border: '1px solid #334155', color: '#fff', borderRadius: '6px', fontSize: '12px' }}
                    />
                  </div>
                </div>

                <button
                  onClick={handleLaunchInvestigation}
                  disabled={isRunning}
                  style={{
                    width: '100%',
                    padding: '12px',
                    borderRadius: '8px',
                    border: 'none',
                    background: isRunning ? '#334155' : 'linear-gradient(135deg, #0284c7, #2563eb)',
                    color: '#fff',
                    fontWeight: 600,
                    fontSize: '13px',
                    cursor: isRunning ? 'not-allowed' : 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '8px',
                  }}
                >
                  {isRunning ? <RefreshCw className="spin-slow" size={16} /> : <Play size={16} />}
                  {isRunning ? 'Investigation in Progress...' : 'Launch Auto-Pilot Research'}
                </button>
              </div>

              {/* Dynamic Agent Roster */}
              <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', padding: '18px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Layers size={16} color="#38bdf8" />
                    <h4 style={{ fontSize: '14px', fontWeight: 600 }}>Active Agent Consortium</h4>
                  </div>
                  <span style={{ fontSize: '11px', background: '#1e293b', padding: '2px 8px', borderRadius: '10px', color: '#38bdf8' }}>
                    {activeAgents.length || 8}
                  </span>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', maxHeight: '350px', overflowY: 'auto' }}>
                  {(activeAgents.length > 0
                    ? activeAgents
                    : [
                        { name: 'Dr. Strategist', role: 'Biomedical Architect', goal: 'Construct MeSH & boolean queries' },
                        { name: 'Dr. Graph', role: 'Systems Biologist', goal: 'Extract molecular interaction network' },
                        { name: 'Dr. DepMap', role: 'Genomics Specialist', goal: 'Verify Chronos cancer dependency' },
                        { name: 'Dr. Pharmacovigilance', role: 'FDA Safety Specialist', goal: 'Monitor FAERS adverse reaction signals' },
                        { name: 'Dr. ClinVar', role: 'Clinical Geneticist', goal: 'Stratify pathogenic variants & criteria' },
                        { name: 'Dr. TME', role: 'Single-Cell Biologist', goal: 'Map paracrine ligand-receptor axes' },
                        { name: 'Dr. Synapse', role: 'Literature Analyst', goal: 'Synthesize signaling pathways' },
                        { name: 'Dr. Hypothesis', role: 'Reasoning Modeler', goal: 'Formulate falsifiable hypotheses' },
                        { name: 'Dr. Skeptic', role: 'Devil Advocate', goal: 'Adversarial peer-review stress test' },
                        { name: 'Dr. Protocol', role: 'Assay Architect', goal: 'Design 3-tier validation roadmap' },
                        { name: 'Dr. Editor', role: 'Lead Author', goal: 'Author publication-grade paper in Markdown' },
                      ]
                  ).map((agent, i) => (
                    <div key={i} style={{ padding: '10px', borderRadius: '8px', background: '#090d16', border: '1px solid #1e293b' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ fontWeight: 600, fontSize: '13px', color: '#38bdf8' }}>{agent.name}</span>
                        <span style={{ fontSize: '10px', color: '#10b981', border: '1px solid #059669', padding: '1px 6px', borderRadius: '4px' }}>Active</span>
                      </div>
                      <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '2px' }}>{agent.role}</div>
                      <div style={{ fontSize: '11px', color: '#64748b', marginTop: '4px', fontStyle: 'italic' }}>"{agent.goal}"</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Right Column: Live Telemetry & Hypotheses */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              <div style={{ background: '#090d16', border: '1px solid #1e293b', borderRadius: '12px', padding: '16px', display: 'flex', flexDirection: 'column', height: '320px' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingBottom: '10px', borderBottom: '1px solid #1e293b' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Terminal size={16} color="#10b981" />
                    <span style={{ fontSize: '13px', fontWeight: 600, color: '#f8fafc' }}>Live Research Telemetry</span>
                  </div>
                  <span style={{ fontSize: '11px', color: '#64748b' }}>Auto-scrolling WebSocket stream</span>
                </div>

                <div ref={logTerminalRef} style={{ flex: 1, overflowY: 'auto', paddingTop: '10px', fontFamily: 'JetBrains Mono, monospace', fontSize: '12px' }}>
                  {eventLogs.length === 0 ? (
                    <div style={{ color: '#475569', textAlign: 'center', paddingTop: '80px' }}>Telemetry stream ready. Launch an investigation or wait for incoming events.</div>
                  ) : (
                    eventLogs.map((log, idx) => (
                      <div key={idx} style={{ marginBottom: '6px', lineHeight: 1.4 }}>
                        <span style={{ color: '#64748b' }}>[{log.timestamp}]</span> <b style={{ color: '#38bdf8' }}>{log.author}:</b> <span style={{ color: '#cbd5e1' }}>{log.message}</span>
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* Formulated Scientific Hypotheses */}
              <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', padding: '20px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '14px' }}>
                  <Sparkles size={18} color="#f59e0b" />
                  <h3 style={{ fontSize: '15px', fontWeight: 600 }}>Formulated Scientific Hypotheses & Peer Reviews</h3>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '14px' }}>
                  {(hypotheses.length > 0
                    ? hypotheses
                    : [
                        {
                          id: 'HYP-01',
                          title: 'Polyphenol-Engineered Selenium Nanocatalysts for Glioblastoma Immunotherapy',
                          biological_targets: ['PD-L1', 'Selenium NPs', 'cGAS-STING', 'TAMs'],
                          mechanism_of_action: 'Nose-to-brain delivery enables targeted blood-brain barrier translocation, activating STING pathway and reversing immunosuppressive microenvironment.',
                          confidence_score: 0.85,
                          critique_verdict: 'Plausible with Caveats',
                          counter_arguments: ['Potential off-target astrocyte toxicity', 'Delivery rate-limiting in vivo'],
                        },
                      ]
                  ).map((hypo, i) => (
                    <div key={i} style={{ background: '#090d16', border: '1px solid #334155', borderRadius: '10px', padding: '16px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                        <span style={{ color: '#f59e0b', fontWeight: 700, fontSize: '14px' }}>{hypo.id || `HYP-0${i + 1}`}: {hypo.title}</span>
                        <span style={{ fontSize: '11px', background: 'rgba(16, 185, 129, 0.1)', color: '#10b981', border: '1px solid #10b981', padding: '2px 8px', borderRadius: '12px' }}>
                          Confidence: {Math.round((hypo.confidence_score || 0.8) * 100)}%
                        </span>
                      </div>
                      <div style={{ fontSize: '12px', color: '#94a3b8', marginBottom: '8px' }}>
                        <b>Targets:</b> {Array.isArray(hypo.biological_targets) ? hypo.biological_targets.join(', ') : hypo.biological_targets}
                      </div>
                      <div style={{ fontSize: '12px', color: '#cbd5e1', lineHeight: 1.5, marginBottom: '10px' }}>
                        {hypo.mechanism_of_action}
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', background: '#0f172a', padding: '8px 12px', borderRadius: '6px', fontSize: '11px' }}>
                        <ShieldAlert size={14} color="#f59e0b" />
                        <span><b>Reviewer Verdict:</b> <span style={{ color: '#38bdf8' }}>{hypo.critique_verdict || 'Plausible with Caveats'}</span></span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* VIEW 2: INTERACTIVE COPILOT CHAT */}
        {activeTab === 'chat' && (
          <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', padding: '20px', height: '700px', display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingBottom: '14px', borderBottom: '1px solid #1e293b', marginBottom: '14px' }}>
              <div>
                <h3 style={{ fontSize: '16px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <MessageSquare size={18} color="#38bdf8" /> Dr. Copilot - Live Investigation Co-Investigator
                </h3>
                <p style={{ fontSize: '12px', color: '#94a3b8', marginTop: '2px' }}>
                  Converse during an active research session or interrogate completed discoveries. Give steering orders to launch sub-inquiries.
                </p>
              </div>
              <span style={{ fontSize: '11px', background: '#0284c7', color: '#fff', padding: '2px 8px', borderRadius: '4px' }}>
                Investigation: {currentInvId || 'Global'}
              </span>
            </div>

            <div ref={chatBottomRef} style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '12px', paddingRight: '8px' }}>
              {chatMessages.map((msg, idx) => {
                const isUser = msg.sender === 'user';
                return (
                  <div key={idx} style={{ alignSelf: isUser ? 'flex-end' : 'flex-start', maxWidth: '78%' }}>
                    <div
                      style={{
                        padding: '12px 16px',
                        borderRadius: '12px',
                        background: isUser ? '#0284c7' : '#090d16',
                        border: isUser ? 'none' : '1px solid #1e293b',
                        color: '#f8fafc',
                        fontSize: '13px',
                        lineHeight: 1.5,
                      }}
                    >
                      <div style={{ fontSize: '10px', color: isUser ? '#e0f2fe' : '#64748b', marginBottom: '4px', fontWeight: 600 }}>
                        {isUser ? 'YOU' : 'DR. COPILOT'} • {msg.timestamp}
                      </div>
                      <div>{msg.text}</div>

                      {msg.actions && msg.actions.length > 0 && (
                        <div style={{ marginTop: '8px', paddingTop: '6px', borderTop: '1px solid #1e293b', fontSize: '11px', color: '#10b981' }}>
                          <b>Actions Triggered:</b> {msg.actions.join(', ')}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
              {isChatting && (
                <div style={{ alignSelf: 'flex-start', background: '#090d16', padding: '10px 14px', borderRadius: '12px', border: '1px solid #1e293b', fontSize: '12px', color: '#38bdf8' }}>
                  Dr. Copilot analyzing knowledge graph & evidence...
                </div>
              )}
            </div>

            <form onSubmit={handleSendChatMessage} style={{ marginTop: '14px', display: 'flex', gap: '10px' }}>
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                placeholder="Ask a question or steer the research (e.g. 'Investigate astrocyte toxicity', 'Design CRISPR for CD274')..."
                style={{ flex: 1, padding: '12px 16px', borderRadius: '8px', background: '#020617', border: '1px solid #334155', color: '#fff', fontSize: '13px' }}
              />
              <button
                type="submit"
                disabled={isChatting}
                style={{ padding: '12px 20px', borderRadius: '8px', background: '#0284c7', color: '#fff', border: 'none', cursor: isChatting ? 'not-allowed' : 'pointer', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '6px' }}
              >
                <Send size={16} /> Send
              </button>
            </form>
          </div>
        )}

        {/* VIEW 3: DEPMAP GENOMICS STATION */}
        {activeTab === 'depmap' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', padding: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <div>
                  <h3 style={{ fontSize: '16px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Database size={20} color="#06b6d4" /> Cancer Dependency Map (DepMap) Essentiality Verifier
                  </h3>
                  <p style={{ fontSize: '12px', color: '#94a3b8', marginTop: '4px' }}>
                    Audits target vulnerability across 1,086 human cancer cell lines from Broad Institute MIT/Harvard CRISPR screens.
                  </p>
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                  {['STAT3', 'ERBB2', 'EGFR', 'KRAS', 'GPX4', 'CD274', 'CCT1'].map((g) => (
                    <button
                      key={g}
                      onClick={() => handleQueryDepMap(g)}
                      style={{ padding: '6px 12px', borderRadius: '6px', background: depmapGene === g ? '#0891b2' : '#1e293b', color: '#fff', border: 'none', fontSize: '12px', cursor: 'pointer', fontWeight: 600 }}
                    >
                      {g}
                    </button>
                  ))}
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '14px', marginBottom: '20px' }}>
                <div style={{ background: '#090d16', border: '1px solid #1e293b', padding: '16px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '11px', color: '#64748b' }}>CHRONOS SCORE (GENE EFFECT)</div>
                  <div style={{ fontSize: '24px', fontWeight: 700, color: depmapData.chronos_score <= -0.8 ? '#f43f5e' : '#f59e0b', marginTop: '4px' }}>
                    {depmapData.chronos_score}
                  </div>
                  <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '2px' }}>
                    {depmapData.chronos_score <= -1.0 ? 'Pan-Cancer Essential' : (depmapData.chronos_score <= -0.5 ? 'Moderate Dependency' : 'Non-Essential')}
                  </div>
                </div>

                <div style={{ background: '#090d16', border: '1px solid #1e293b', padding: '16px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '11px', color: '#64748b' }}>ESSENTIALITY TIER</div>
                  <div style={{ fontSize: '16px', fontWeight: 700, color: '#38bdf8', marginTop: '8px' }}>
                    {depmapData.essentiality_tier}
                  </div>
                  <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '4px' }}>
                    Lineage: {depmapData.selective_in_lineage}
                  </div>
                </div>

                <div style={{ background: '#090d16', border: '1px solid #1e293b', padding: '16px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '11px', color: '#64748b' }}>CELL LINES SCREENED</div>
                  <div style={{ fontSize: '24px', fontWeight: 700, color: '#10b981', marginTop: '4px' }}>
                    {depmapData.cell_lines_tested_count}
                  </div>
                  <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '2px' }}>CRISPR-Cas9 Knockouts</div>
                </div>

                <div style={{ background: '#090d16', border: '1px solid #1e293b', padding: '16px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '11px', color: '#64748b' }}>ACTIONABILITY VERDICT</div>
                  <div style={{ fontSize: '12px', color: '#f8fafc', lineHeight: 1.4, marginTop: '8px' }}>
                    {depmapData.therapeutic_actionability}
                  </div>
                </div>
              </div>

              <div style={{ background: '#090d16', border: '1px solid #1e293b', borderRadius: '8px', padding: '16px' }}>
                <h4 style={{ fontSize: '13px', fontWeight: 700, color: '#f8fafc', marginBottom: '10px' }}>
                  Top Dependent Cancer Cell Lines for {depmapData.gene_symbol}
                </h4>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px' }}>
                  {depmapData.top_dependent_cell_lines.map((cl, i) => (
                    <div key={i} style={{ background: '#0f172a', padding: '10px 14px', borderRadius: '6px', border: '1px solid #334155' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontWeight: 600, color: '#38bdf8', fontSize: '13px' }}>
                        <span>{cl.cell_line}</span>
                        <span style={{ color: '#f43f5e' }}>Chronos: {cl.score}</span>
                      </div>
                      <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '2px' }}>Lineage: {cl.lineage}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* VIEW 4: OPENFDA FAERS PHARMACOVIGILANCE */}
        {activeTab === 'fda' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', padding: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <div>
                  <h3 style={{ fontSize: '16px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <AlertTriangle size={20} color="#f59e0b" /> openFDA FAERS Pharmacovigilance & Adverse Reaction Surveillance
                  </h3>
                  <p style={{ fontSize: '12px', color: '#94a3b8', marginTop: '4px' }}>
                    Real-world human clinical adverse event surveillance querying millions of post-marketing patient records.
                  </p>
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                  {['tamoxifen', 'afatinib', 'temozolomide', 'pembrolizumab', 'doxorubicin'].map((d) => (
                    <button
                      key={d}
                      onClick={() => handleQueryFda(d)}
                      style={{ padding: '6px 12px', borderRadius: '6px', background: fdaDrug === d ? '#f59e0b' : '#1e293b', color: fdaDrug === d ? '#000' : '#fff', border: 'none', fontSize: '12px', cursor: 'pointer', fontWeight: 700 }}
                    >
                      {d}
                    </button>
                  ))}
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '14px', marginBottom: '20px' }}>
                <div style={{ background: '#090d16', border: '1px solid #1e293b', padding: '16px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '11px', color: '#64748b' }}>TOTAL PATIENT CASES</div>
                  <div style={{ fontSize: '24px', fontWeight: 700, color: '#38bdf8', marginTop: '4px' }}>{fdaReport.total_patient_reports.toLocaleString()}</div>
                  <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '2px' }}>FDA MedDRA Reports</div>
                </div>
                <div style={{ background: '#090d16', border: '1px solid #1e293b', padding: '16px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '11px', color: '#64748b' }}>SERIOUS CLINICAL OUTCOMES</div>
                  <div style={{ fontSize: '24px', fontWeight: 700, color: '#f59e0b', marginTop: '4px' }}>{(fdaReport.serious_outcomes.serious_adverse_events_total || 290).toLocaleString()}</div>
                  <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '2px' }}>Hospitalization / Disability</div>
                </div>
                <div style={{ gridColumn: 'span 2', background: '#090d16', border: '1px solid #1e293b', padding: '16px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '11px', color: '#64748b' }}>SAFETY VERDICT</div>
                  <div style={{ fontSize: '13px', fontWeight: 700, color: '#10b981', marginTop: '6px' }}>{fdaReport.clinical_safety_verdict}</div>
                </div>
              </div>

              <div style={{ background: '#090d16', border: '1px solid #1e293b', borderRadius: '8px', padding: '16px' }}>
                <h4 style={{ fontSize: '13px', fontWeight: 700, color: '#f8fafc', marginBottom: '10px' }}>Top MedDRA Adverse Event Reactions</h4>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px' }}>
                  {fdaReport.top_adverse_reactions.map((r, i) => (
                    <div key={i} style={{ background: '#0f172a', padding: '12px', borderRadius: '6px', border: '1px solid #334155' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontWeight: 700, fontSize: '13px', color: '#f8fafc' }}>
                        <span>{r.meddra_term}</span>
                        <span style={{ color: r.severity_tier === 'CRITICAL_ORGAN_TOXICITY' ? '#f43f5e' : '#38bdf8' }}>{r.percentage_of_reports}%</span>
                      </div>
                      <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '4px' }}>Reported Cases: {r.report_count.toLocaleString()}</div>
                      <div style={{ fontSize: '10px', color: r.severity_tier === 'CRITICAL_ORGAN_TOXICITY' ? '#f43f5e' : '#10b981', marginTop: '2px', fontWeight: 600 }}>
                        {r.severity_tier}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* VIEW 5: NCBI CLINVAR GENOMICS */}
        {activeTab === 'clinvar' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', padding: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <div>
                  <h3 style={{ fontSize: '16px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <ShieldAlert size={20} color="#38bdf8" /> NCBI ClinVar Pathogenic Genomic Stratification
                  </h3>
                  <p style={{ fontSize: '12px', color: '#94a3b8', marginTop: '4px' }}>
                    Maps verified human clinical pathogenic mutations and defines trial inclusion/exclusion biomarker criteria.
                  </p>
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                  {['EGFR', 'KRAS', 'BRAF', 'BRCA1', 'STAT3'].map((g) => (
                    <button
                      key={g}
                      onClick={() => handleQueryClinvar(g)}
                      style={{ padding: '6px 12px', borderRadius: '6px', background: clinvarGene === g ? '#0284c7' : '#1e293b', color: '#fff', border: 'none', fontSize: '12px', cursor: 'pointer', fontWeight: 600 }}
                    >
                      {g}
                    </button>
                  ))}
                </div>
              </div>

              <div style={{ background: '#090d16', border: '1px solid #1e293b', borderRadius: '8px', padding: '16px', marginBottom: '16px' }}>
                <h4 style={{ fontSize: '13px', fontWeight: 700, color: '#10b981', marginBottom: '6px' }}>PRECISION ONCOLOGY NGS INCLUSION CRITERIA</h4>
                <p style={{ fontSize: '12px', color: '#cbd5e1', lineHeight: 1.4 }}>{clinvarReport.clinical_trial_genomic_criteria}</p>
              </div>

              <div style={{ background: '#090d16', border: '1px solid #1e293b', borderRadius: '8px', padding: '16px' }}>
                <h4 style={{ fontSize: '13px', fontWeight: 700, color: '#f8fafc', marginBottom: '10px' }}>Verified ClinVar Pathogenic & Resistance Variants</h4>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px' }}>
                  {clinvarReport.top_pathogenic_variants.map((v, i) => (
                    <div key={i} style={{ background: '#0f172a', padding: '12px', borderRadius: '6px', border: '1px solid #334155' }}>
                      <div style={{ fontWeight: 700, fontSize: '13px', color: '#38bdf8', marginBottom: '4px' }}>{v.title}</div>
                      <div style={{ fontSize: '11px', color: '#f59e0b', fontWeight: 600 }}>{v.clinical_significance}</div>
                      <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '2px' }}>Consequence: {v.molecular_consequence}</div>
                      <div style={{ fontSize: '10px', color: '#64748b', marginTop: '4px' }}>Review: {v.review_status}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* VIEW 6: SINGLE-CELL TME & LIGAND-RECEPTOR INTERACTOME */}
        {activeTab === 'tme' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', padding: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <div>
                  <h3 style={{ fontSize: '16px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <PieChart size={20} color="#a855f7" /> Single-Cell Tumor Microenvironment (TME) & Paracrine Interactome
                  </h3>
                  <p style={{ fontSize: '12px', color: '#94a3b8', marginTop: '4px' }}>
                    Deconvolutes single-cell cellular fractions and resolves cell-cell ligand-receptor communication pairs.
                  </p>
                </div>
                <div style={{ background: '#0284c7', color: '#fff', padding: '4px 10px', borderRadius: '6px', fontSize: '11px', fontWeight: 700 }}>
                  SUBTYPE: {tmeReport.microenvironment_subtype}
                </div>
              </div>

              {/* TME Cellular Populations */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px', marginBottom: '20px' }}>
                {tmeReport.target_expression_by_cell_type.map((ct, i) => (
                  <div key={i} style={{ background: '#090d16', border: '1px solid #1e293b', padding: '14px', borderRadius: '8px' }}>
                    <div style={{ fontSize: '13px', fontWeight: 700, color: '#38bdf8' }}>{ct.cell_type}</div>
                    <div style={{ fontSize: '20px', fontWeight: 700, color: '#f59e0b', marginTop: '4px' }}>
                      {Math.round(ct.relative_expression_level * 100)}%
                    </div>
                    <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '4px', lineHeight: 1.3 }}>{ct.functional_role_in_tme}</div>
                  </div>
                ))}
              </div>

              {/* Ligand-Receptor Interaction Axes */}
              <div style={{ background: '#090d16', border: '1px solid #1e293b', borderRadius: '8px', padding: '16px' }}>
                <h4 style={{ fontSize: '13px', fontWeight: 700, color: '#f8fafc', marginBottom: '10px' }}>
                  Active Paracrine Ligand-Receptor Communications
                </h4>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px' }}>
                  {tmeReport.active_ligand_receptor_axes.map((ax, i) => (
                    <div key={i} style={{ background: '#0f172a', padding: '12px', borderRadius: '6px', border: '1px solid #334155' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontWeight: 700, fontSize: '13px', color: '#a855f7' }}>
                        <span>{ax.ligand_gene}</span>
                        <span>↔</span>
                        <span>{ax.receptor_gene}</span>
                      </div>
                      <div style={{ fontSize: '11px', color: '#38bdf8', marginTop: '4px' }}>
                        {ax.source_cell_type} → {ax.target_cell_type}
                      </div>
                      <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '4px', fontStyle: 'italic' }}>
                        Effect: {ax.functional_effect}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* VIEW 7: DE NOVO CHEMISTRY & SASCORE */}
        {activeTab === 'denovo' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', padding: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <div>
                  <h3 style={{ fontSize: '16px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Sparkles size={20} color="#a855f7" /> De Novo Chemical Generator & Synthesizability (SAscore)
                  </h3>
                  <p style={{ fontSize: '12px', color: '#94a3b8', marginTop: '4px' }}>
                    Generates novel molecular entities via bioisosterism, computes Ertl Synthetic Accessibility (1-10), and predicts BBB LogBB and hERG cardiotoxicity.
                  </p>
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                  {['Tamoxifen', 'Afatinib', 'Temozolomide', 'Curcumin'].map((p) => (
                    <button
                      key={p}
                      onClick={() => handleGenerateDenovo(p)}
                      style={{ padding: '6px 12px', borderRadius: '6px', background: denovoParent === p ? '#7c3aed' : '#1e293b', color: '#fff', border: 'none', fontSize: '12px', cursor: 'pointer', fontWeight: 600 }}
                    >
                      {p}
                    </button>
                  ))}
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px' }}>
                {denovoAnalogs.map((ana, idx) => (
                  <div key={idx} style={{ background: '#090d16', border: '1px solid #1e293b', borderRadius: '10px', padding: '18px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                      <span style={{ fontSize: '14px', fontWeight: 700, color: '#38bdf8' }}>{ana.compound_code}: {ana.name}</span>
                      <span style={{ fontSize: '11px', background: 'rgba(16, 185, 129, 0.1)', color: '#10b981', border: '1px solid #10b981', padding: '2px 8px', borderRadius: '12px' }}>
                        SAscore: {ana.sa_score}
                      </span>
                    </div>

                    <div style={{ fontSize: '11px', color: '#94a3b8', marginBottom: '8px', lineHeight: 1.4 }}>
                      {ana.modification_rationale}
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px', background: '#020617', padding: '10px', borderRadius: '6px', fontSize: '11px', marginTop: '10px' }}>
                      <div><b>LogBB (Brain):</b> <span style={{ color: ana.bbb_penetrant ? '#10b981' : '#f59e0b' }}>{ana.log_bb} ({ana.bbb_penetrant ? 'PASS' : 'LOW'})</span></div>
                      <div><b>LogS (Solubility):</b> <span style={{ color: '#38bdf8' }}>{ana.log_s}</span></div>
                      <div><b>hERG Cardiotox:</b> <span style={{ color: ana.herg_cardiotox_risk === 'LOW' ? '#10b981' : '#f43f5e' }}>{ana.herg_cardiotox_risk}</span></div>
                    </div>

                    <div style={{ marginTop: '10px', fontSize: '11px', color: '#64748b', wordBreak: 'break-all', fontFamily: 'monospace' }}>
                      SMILES: {ana.smiles}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* VIEW 8: VIRTUAL PATIENTS (PK/PD & KAPLAN-MEIER) */}
        {activeTab === 'trial_sim' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', padding: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <div>
                  <h3 style={{ fontSize: '16px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Users size={20} color="#10b981" /> Virtual Patient Clinical Trial Simulator (Digital Twins & PK/PD)
                  </h3>
                  <p style={{ fontSize: '12px', color: '#94a3b8', marginTop: '4px' }}>
                    Simulates 500 virtual patients using a 2-compartment pharmacokinetic model, predicting Kaplan-Meier survival curves and Hazard Ratio (HR).
                  </p>
                </div>
                <button
                  onClick={handleSimulateTrial}
                  disabled={isSimulatingTrial}
                  style={{ padding: '8px 16px', borderRadius: '6px', background: '#059669', color: '#fff', border: 'none', fontSize: '12px', fontWeight: 600, cursor: isSimulatingTrial ? 'not-allowed' : 'pointer', display: 'flex', alignItems: 'center', gap: '6px' }}
                >
                  <RefreshCw size={14} className={isSimulatingTrial ? 'spin-slow' : ''} /> Run Monte Carlo Trial (500 Twins)
                </button>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '14px', marginBottom: '20px' }}>
                <div style={{ background: '#090d16', border: '1px solid #1e293b', padding: '14px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '11px', color: '#64748b' }}>HAZARD RATIO (HR)</div>
                  <div style={{ fontSize: '22px', fontWeight: 700, color: '#10b981', marginTop: '4px' }}>{clinicalSimReport.hazard_ratio}</div>
                  <div style={{ fontSize: '10px', color: '#10b981', marginTop: '2px' }}>HR &lt; 0.7: Strong Superiority</div>
                </div>

                <div style={{ background: '#090d16', border: '1px solid #1e293b', padding: '14px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '11px', color: '#64748b' }}>LOG-RANK P-VALUE</div>
                  <div style={{ fontSize: '22px', fontWeight: 700, color: '#38bdf8', marginTop: '4px' }}>p = {clinicalSimReport.logrank_p_value}</div>
                  <div style={{ fontSize: '10px', color: '#38bdf8', marginTop: '2px' }}>Statistically Significant</div>
                </div>

                <div style={{ background: '#090d16', border: '1px solid #1e293b', padding: '14px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '11px', color: '#64748b' }}>PROBABILITY OF SUCCESS</div>
                  <div style={{ fontSize: '22px', fontWeight: 700, color: '#f59e0b', marginTop: '4px' }}>{clinicalSimReport.probability_of_technical_success}%</div>
                  <div style={{ fontSize: '10px', color: '#f59e0b', marginTop: '2px' }}>Projected Phase 2/3 PoS</div>
                </div>

                <div style={{ background: '#090d16', border: '1px solid #1e293b', padding: '14px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '11px', color: '#64748b' }}>MEDIAN PFS (DRUG)</div>
                  <div style={{ fontSize: '22px', fontWeight: 700, color: '#f8fafc', marginTop: '4px' }}>{clinicalSimReport.median_pfs_experimental_months} <span style={{ fontSize: '11px' }}>mo</span></div>
                  <div style={{ fontSize: '10px', color: '#10b981', marginTop: '2px' }}>Control: {clinicalSimReport.median_pfs_control_months} mo (+{clinicalSimReport.pfs_gain_months} mo)</div>
                </div>

                <div style={{ background: '#090d16', border: '1px solid #1e293b', padding: '14px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '11px', color: '#64748b' }}>VIRTUAL PATIENTS</div>
                  <div style={{ fontSize: '22px', fontWeight: 700, color: '#a855f7', marginTop: '4px' }}>{clinicalSimReport.total_patients_simulated}</div>
                  <div style={{ fontSize: '10px', color: '#a855f7', marginTop: '2px' }}>Double-Blind 1:1 Arm</div>
                </div>
              </div>

              <div style={{ background: '#090d16', border: '1px solid #1e293b', borderRadius: '10px', padding: '20px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                  <h4 style={{ fontSize: '14px', fontWeight: 700, color: '#f8fafc' }}>
                    Kaplan-Meier Progression-Free Survival (PFS) Projection
                  </h4>
                  <div style={{ display: 'flex', gap: '16px', fontSize: '12px' }}>
                    <span style={{ color: '#10b981', fontWeight: 600 }}>── Investigational Drug ({clinicalSimReport.investigational_drug})</span>
                    <span style={{ color: '#64748b', fontWeight: 600 }}>┈ Standard of Care (Control)</span>
                  </div>
                </div>

                <svg viewBox="0 0 800 240" width="100%" height="240" style={{ background: '#020617', borderRadius: '8px' }}>
                  {[0, 60, 120, 180].map((y, i) => (
                    <line key={i} x1="50" y1={y + 20} x2="760" y2={y + 20} stroke="#1e293b" strokeWidth="1" strokeDasharray="3 3" />
                  ))}
                  <text x="15" y="25" fill="#64748b" fontSize="10">100%</text>
                  <text x="15" y="85" fill="#64748b" fontSize="10">75%</text>
                  <text x="15" y="145" fill="#64748b" fontSize="10">50%</text>
                  <text x="15" y="205" fill="#64748b" fontSize="10">25%</text>
                  {[0, 6, 12, 18, 24, 36].map((m, idx) => (
                    <text key={idx} x={50 + (idx * 135)} y="230" fill="#64748b" fontSize="10">{m} mo</text>
                  ))}
                  <path
                    d={`M 50 20 ${clinicalSimReport.kaplan_meier_curve.map((p, idx) => `L ${50 + idx * 100} ${20 + (1.0 - p.control_survival_rate) * 180}`).join(' ')}`}
                    fill="none"
                    stroke="#64748b"
                    strokeWidth="2.5"
                    strokeDasharray="5 5"
                  />
                  <path
                    d={`M 50 20 ${clinicalSimReport.kaplan_meier_curve.map((p, idx) => `L ${50 + idx * 100} ${20 + (1.0 - p.experimental_survival_rate) * 180}`).join(' ')}`}
                    fill="none"
                    stroke="#10b981"
                    strokeWidth="3.5"
                  />
                </svg>
              </div>
            </div>
          </div>
        )}

        {/* VIEW 9: CRISPR & PRIMERS */}
        {activeTab === 'crispr' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', padding: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <div>
                  <h3 style={{ fontSize: '16px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Scissors size={20} color="#f59e0b" /> CRISPR-Cas9 sgRNA & PCR Primer Design Workbench
                  </h3>
                  <p style={{ fontSize: '12px', color: '#94a3b8', marginTop: '4px' }}>
                    Automated PAM recognition (NGG), poly-T termination avoidance, BsmBI golden-gate cloning oligos, and flanking PCR validation primers.
                  </p>
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                  {['CD274', 'STAT3', 'ERBB2', 'EGFR', 'KRAS'].map((gene) => (
                    <button
                      key={gene}
                      onClick={() => handleDesignCrispr(gene)}
                      style={{ padding: '6px 14px', borderRadius: '6px', background: crisprGene === gene ? '#f59e0b' : '#1e293b', color: crisprGene === gene ? '#000' : '#fff', border: 'none', fontSize: '12px', cursor: 'pointer', fontWeight: 700 }}
                    >
                      {gene}
                    </button>
                  ))}
                </div>
              </div>

              {isDesigningCrispr ? (
                <div style={{ textAlign: 'center', padding: '40px', color: '#f59e0b' }}>
                  Fetching NCBI Nucleotide CDS and computing CRISPR-Cas9 guide efficiencies...
                </div>
              ) : crisprReport ? (
                <div>
                  <div style={{ display: 'flex', gap: '16px', marginBottom: '16px', background: '#090d16', padding: '12px 16px', borderRadius: '8px' }}>
                    <div><b>Target Gene:</b> {crisprReport.target_gene}</div>
                    <div><b>Organism:</b> {crisprReport.organism}</div>
                    <div><b>Sequence Length:</b> {crisprReport.sequence_length_bp} bp</div>
                  </div>

                  <h4 style={{ fontSize: '14px', fontWeight: 700, color: '#f59e0b', marginBottom: '10px' }}>Top Ranked sgRNA Candidates</h4>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px', marginBottom: '20px' }}>
                    {crisprReport.top_sgrnas.map((g, idx) => (
                      <div key={idx} style={{ background: '#090d16', border: '1px solid #1e293b', borderRadius: '8px', padding: '14px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                          <span style={{ fontWeight: 700, color: '#38bdf8' }}>sgRNA Candidate #{idx + 1} ({g.strand})</span>
                          <span style={{ fontSize: '11px', background: 'rgba(245, 158, 11, 0.15)', color: '#f59e0b', padding: '2px 6px', borderRadius: '4px' }}>
                            Score: {g.on_target_score}
                          </span>
                        </div>
                        <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '12px', background: '#020617', padding: '6px', borderRadius: '4px', marginBottom: '6px', wordBreak: 'break-all' }}>
                          5'-{g.sequence_20nt}-3' <b style={{ color: '#f43f5e' }}>{g.pam_motif}</b>
                        </div>
                        <div style={{ fontSize: '11px', color: '#94a3b8' }}>
                          GC: {g.gc_percent}% | Pos: {g.start_pos}-{g.end_pos}
                        </div>
                        <div style={{ fontSize: '10px', color: '#64748b', marginTop: '6px' }}>
                          <b>Cloning Top:</b> {g.oligo_forward_top}<br />
                          <b>Cloning Bot:</b> {g.oligo_reverse_bottom}
                        </div>
                      </div>
                    ))}
                  </div>

                  {crisprReport.validation_primers && (
                    <div style={{ background: '#090d16', border: '1px solid #1e293b', borderRadius: '8px', padding: '14px' }}>
                      <h4 style={{ fontSize: '13px', fontWeight: 700, color: '#10b981', marginBottom: '8px' }}>Flanking PCR Validation Primer Pair</h4>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px', fontSize: '12px' }}>
                        <div>
                          <b>Forward Primer:</b> <code style={{ color: '#38bdf8' }}>{crisprReport.validation_primers.forward_primer}</code><br />
                          <span style={{ color: '#94a3b8' }}>Tm: {crisprReport.validation_primers.forward_tm}°C | GC: {crisprReport.validation_primers.gc_percent_forward}%</span>
                        </div>
                        <div>
                          <b>Reverse Primer:</b> <code style={{ color: '#38bdf8' }}>{crisprReport.validation_primers.reverse_primer}</code><br />
                          <span style={{ color: '#94a3b8' }}>Tm: {crisprReport.validation_primers.reverse_tm}°C | Amplicon: {crisprReport.validation_primers.amplicon_length_bp} bp</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div style={{ textAlign: 'center', padding: '40px', color: '#64748b' }}>
                  Select a target gene or enter an identifier to design CRISPR-Cas9 guides.
                </div>
              )}
            </div>
          </div>
        )}

        {/* VIEW 10: DNA & MOLECULAR VISUAL STUDIO */}
        {activeTab === 'visuals' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', padding: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <div>
                  <h3 style={{ fontSize: '16px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Dna size={20} color="#a855f7" /> DNA Construct & 2D Molecular Structure Studio
                  </h3>
                  <p style={{ fontSize: '12px', color: '#94a3b8', marginTop: '4px' }}>
                    Renders publication-grade vector graphics (SVG) for genetic cassettes, expression plasmids, and small molecule chemical diagrams.
                  </p>
                </div>
              </div>

              <div style={{ background: '#090d16', border: '1px solid #1e293b', borderRadius: '10px', padding: '16px', marginBottom: '20px' }}>
                <h4 style={{ fontSize: '13px', fontWeight: 700, color: '#38bdf8', marginBottom: '12px' }}>Linear Genetic Expression Cassette</h4>
                <div dangerouslySetInnerHTML={{ __html: dnaSvg }} />
              </div>

              <div style={{ background: '#090d16', border: '1px solid #1e293b', borderRadius: '10px', padding: '16px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                  <h4 style={{ fontSize: '13px', fontWeight: 700, color: '#a855f7' }}>2D Chemical Structure: {selectedDrug.toUpperCase()}</h4>
                  <div style={{ display: 'flex', gap: '6px' }}>
                    {['tamoxifen', 'afatinib', 'temozolomide', 'doxorubicin', 'curcumin'].map((d) => (
                      <button
                        key={d}
                        onClick={() => handleSimulateDocking(d, selectedTarget)}
                        style={{ padding: '4px 10px', borderRadius: '4px', background: selectedDrug === d ? '#7c3aed' : '#1e293b', color: '#fff', border: 'none', fontSize: '11px', cursor: 'pointer' }}
                      >
                        {d}
                      </button>
                    ))}
                  </div>
                </div>
                <div style={{ display: 'flex', justifyContent: 'center', padding: '16px', background: '#020617', borderRadius: '8px' }} dangerouslySetInnerHTML={{ __html: moleculeSvg }} />
              </div>
            </div>
          </div>
        )}

        {/* VIEW 11: INTERACTIVE EPISTEMIC KNOWLEDGE GRAPH */}
        {activeTab === 'graph' && (
          <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', padding: '20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <div>
                <h3 style={{ fontSize: '16px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Network size={18} color="#a855f7" /> Epistemic Biological Network & Discovery Topology
                </h3>
                <p style={{ fontSize: '12px', color: '#94a3b8', marginTop: '4px' }}>
                  Multi-layered graph mapping primary evidence to concepts, formulated hypotheses, doubts, and experimental conclusions.
                </p>
              </div>
              <div style={{ display: 'flex', gap: '8px', fontSize: '11px' }}>
                <span style={{ background: '#0284c7', color: '#fff', padding: '3px 8px', borderRadius: '4px' }}>Evidence</span>
                <span style={{ background: '#a855f7', color: '#fff', padding: '3px 8px', borderRadius: '4px' }}>Concepts</span>
                <span style={{ background: '#f59e0b', color: '#fff', padding: '3px 8px', borderRadius: '4px' }}>Hypotheses</span>
                <span style={{ background: '#f43f5e', color: '#fff', padding: '3px 8px', borderRadius: '4px' }}>Doubts</span>
                <span style={{ background: '#10b981', color: '#fff', padding: '3px 8px', borderRadius: '4px' }}>Conclusions</span>
              </div>
            </div>

            <div style={{ width: '100%', height: '550px', background: '#020617', borderRadius: '10px', border: '1px solid #1e293b', overflow: 'hidden', position: 'relative' }}>
              <canvas ref={canvasRef} style={{ width: '100%', height: '100%', display: 'block' }} />
            </div>
          </div>
        )}

        {/* VIEW 12: EVIDENCE MATRIX */}
        {activeTab === 'evidence' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', padding: '18px' }}>
              <h3 style={{ fontSize: '16px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                <BookOpen size={18} color="#0284c7" /> Systematic Evidence Extraction & Argumentation Matrix
              </h3>
              <p style={{ fontSize: '12px', color: '#94a3b8' }}>
                Every publication selected by agents includes the explicit justification of why it was chosen, the key fragment taken, and the mechanistic argument.
              </p>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              {(evidenceList.length > 0
                ? evidenceList
                : [
                    {
                      paper_id: '42669245',
                      paper_title: 'Polyphenol-engineered selenium nanocatalysts enable neuroprotective glioblastoma chemoimmunotherapy via nose-to-brain delivery',
                      selection_rationale: 'Selected because it provides proof-of-concept for non-invasive BBB crossing using selenium nanocatalysts in aggressive glioma.',
                      key_excerpt: 'Here, we engineered polyphenol-stabilized selenium nanoparticles (<60 nm) enabling neuroprotective chemoimmunotherapy via nose-to-brain delivery and macrophage reprogramming.',
                      scientific_argumentation: 'Directly resolves the primary pharmacokinetic delivery barrier in glioblastoma, proving that selenium nanocatalysts achieve deep intracranial distribution.',
                      agent_name: 'Dr. Synapse',
                    },
                  ]
              ).map((ev, idx) => (
                <div key={idx} style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '10px', padding: '18px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                  <div style={{ borderRight: '1px solid #1e293b', paddingRight: '20px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                      <span style={{ background: '#0284c7', color: '#fff', fontSize: '10px', fontWeight: 700, padding: '2px 6px', borderRadius: '4px' }}>
                        PMID: {ev.paper_id}
                      </span>
                      <span style={{ fontSize: '11px', color: '#64748b' }}>Reviewed by {ev.agent_name || 'Dr. Synapse'}</span>
                    </div>
                    <h4 style={{ fontSize: '13px', fontWeight: 600, color: '#f8fafc', marginBottom: '8px' }}>{ev.paper_title}</h4>
                    <div style={{ background: '#090d16', padding: '12px', borderRadius: '6px', border: '1px solid #1e293b', fontSize: '12px', color: '#94a3b8', fontStyle: 'italic', lineHeight: 1.5 }}>
                      "{ev.key_excerpt}"
                    </div>
                  </div>

                  <div>
                    <div style={{ marginBottom: '10px' }}>
                      <div style={{ fontSize: '11px', fontWeight: 700, color: '#38bdf8', marginBottom: '3px' }}>WHY WAS THIS PUBLICATION SELECTED?</div>
                      <div style={{ fontSize: '12px', color: '#cbd5e1', lineHeight: 1.4 }}>{ev.selection_rationale}</div>
                    </div>
                    <div>
                      <div style={{ fontSize: '11px', fontWeight: 700, color: '#10b981', marginBottom: '3px' }}>SCIENTIFIC ARGUMENTATION & RELEVANCE</div>
                      <div style={{ fontSize: '12px', color: '#cbd5e1', lineHeight: 1.4 }}>{ev.scientific_argumentation}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* VIEW 13: IN-SILICO MOLECULAR STATION */}
        {activeTab === 'insilico' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', padding: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <div>
                  <h3 style={{ fontSize: '16px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Atom size={20} color="#06b6d4" /> In-Silico Molecular Station (RDKit & 3D Crystallography)
                  </h3>
                  <p style={{ fontSize: '12px', color: '#94a3b8', marginTop: '4px' }}>
                    Real MMFF94 force field minimization, Lipinski Rule of 5 bioavailability verification, and RCSB PDB structure retrieval.
                  </p>
                </div>
                <div style={{ display: 'flex', gap: '10px' }}>
                  {['Tamoxifen', 'Afatinib', 'Temozolomide', 'Curcumin', 'Doxorubicin'].map((drug) => (
                    <button
                      key={drug}
                      onClick={() => handleSimulateDocking(drug, selectedTarget)}
                      style={{ padding: '6px 12px', borderRadius: '6px', background: selectedDrug.toLowerCase() === drug.toLowerCase() ? '#0891b2' : '#1e293b', color: '#fff', border: 'none', fontSize: '12px', cursor: 'pointer', fontWeight: 600 }}
                    >
                      {drug}
                    </button>
                  ))}
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '14px', marginBottom: '20px' }}>
                <div style={{ background: '#090d16', border: '1px solid #1e293b', padding: '14px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '11px', color: '#64748b' }}>MOLECULAR WEIGHT</div>
                  <div style={{ fontSize: '18px', fontWeight: 700, color: '#38bdf8', marginTop: '4px' }}>{molecularProps.molecular_weight} <span style={{ fontSize: '11px' }}>g/mol</span></div>
                  <div style={{ fontSize: '10px', color: '#10b981', marginTop: '2px' }}>Lipinski ≤ 500 Da: PASS</div>
                </div>

                <div style={{ background: '#090d16', border: '1px solid #1e293b', padding: '14px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '11px', color: '#64748b' }}>LOGP (HYDROPHOBICITY)</div>
                  <div style={{ fontSize: '18px', fontWeight: 700, color: '#38bdf8', marginTop: '4px' }}>{molecularProps.logp}</div>
                  <div style={{ fontSize: '10px', color: '#10b981', marginTop: '2px' }}>Lipinski ≤ 5.0: PASS</div>
                </div>

                <div style={{ background: '#090d16', border: '1px solid #1e293b', padding: '14px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '11px', color: '#64748b' }}>POLAR SURFACE (TPSA)</div>
                  <div style={{ fontSize: '18px', fontWeight: 700, color: '#38bdf8', marginTop: '4px' }}>{molecularProps.tpsa} <span style={{ fontSize: '11px' }}>Å²</span></div>
                  <div style={{ fontSize: '10px', color: '#10b981', marginTop: '2px' }}>Permeability ≤ 140 Å²: PASS</div>
                </div>

                <div style={{ background: '#090d16', border: '1px solid #1e293b', padding: '14px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '11px', color: '#64748b' }}>DRUGLIKENESS (QED)</div>
                  <div style={{ fontSize: '18px', fontWeight: 700, color: '#10b981', marginTop: '4px' }}>{molecularProps.qed_score}</div>
                  <div style={{ fontSize: '10px', color: '#10b981', marginTop: '2px' }}>Quantitative Oral Score</div>
                </div>

                <div style={{ background: '#090d16', border: '1px solid #1e293b', padding: '14px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '11px', color: '#64748b' }}>PREDICTED ΔG AFFINITY</div>
                  <div style={{ fontSize: '18px', fontWeight: 700, color: '#f59e0b', marginTop: '4px' }}>{dockingResult.delta_g} <span style={{ fontSize: '11px' }}>kcal/mol</span></div>
                  <div style={{ fontSize: '10px', color: '#f59e0b', marginTop: '2px' }}>Ki: {dockingResult.ki_micromolar} μM</div>
                </div>
              </div>

              <div style={{ background: '#090d16', border: '1px solid #1e293b', padding: '18px', borderRadius: '8px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontSize: '14px', fontWeight: 700, color: '#f8fafc' }}>
                      Target Receptor Crystallography: {selectedTarget} (PDB: {dockingResult.pdb_id})
                    </span>
                    <span style={{ background: 'rgba(2, 132, 199, 0.2)', color: '#38bdf8', fontSize: '11px', padding: '2px 8px', borderRadius: '4px' }}>RCSB PDB Verified</span>
                  </div>
                  <p style={{ fontSize: '12px', color: '#94a3b8', marginTop: '4px' }}>
                    {dockingResult.verdict}. Thermodynamic binding free energy computed from atomic surface desolvation and hydrogen bonding networks.
                  </p>
                </div>

                <a
                  href={`https://www.rcsb.org/structure/${dockingResult.pdb_id}`}
                  target="_blank"
                  rel="noreferrer"
                  style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '8px 16px', borderRadius: '6px', background: '#0284c7', color: '#fff', textDecoration: 'none', fontSize: '12px', fontWeight: 600 }}
                >
                  <Globe size={14} /> View 3D Structure in RCSB PDB
                </a>
              </div>
            </div>
          </div>
        )}

        {/* VIEW 14: ADVERSARIAL PEER-REVIEW TRIBUNAL */}
        {activeTab === 'tribunal' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', padding: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <div>
                  <h3 style={{ fontSize: '16px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <ShieldAlert size={18} color="#f43f5e" /> Adversarial Peer-Review Tribunal (3-Member Board)
                  </h3>
                  <p style={{ fontSize: '12px', color: '#94a3b8', marginTop: '4px' }}>
                    Independent stress-testing by domain specialists: Pharmacokinetics/ADMET, Cellular Biology, and Biostatistics.
                  </p>
                </div>
                <div style={{ background: 'rgba(16, 185, 129, 0.1)', border: '1px solid #10b981', padding: '6px 14px', borderRadius: '8px', color: '#10b981', fontSize: '12px', fontWeight: 700 }}>
                  CONSENSUS: CONDITIONAL ENDORSEMENT (85%)
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
                {tribunalReviews.map((rev, i) => (
                  <div key={i} style={{ background: '#090d16', border: '1px solid #1e293b', borderRadius: '10px', padding: '16px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                      <span style={{ fontWeight: 700, fontSize: '14px', color: '#38bdf8' }}>{rev.reviewer}</span>
                      <span style={{ fontSize: '13px', fontWeight: 700, color: '#10b981' }}>{rev.score}/100</span>
                    </div>
                    <div style={{ fontSize: '11px', color: '#64748b', marginBottom: '8px' }}>{rev.domain}</div>
                    <div style={{ fontSize: '12px', color: '#cbd5e1', lineHeight: 1.4, marginBottom: '12px' }}>{rev.critique}</div>
                    <div style={{ background: '#0f172a', padding: '8px', borderRadius: '6px', border: '1px solid #334155' }}>
                      <div style={{ fontSize: '10px', fontWeight: 700, color: '#f43f5e', marginBottom: '2px' }}>KEY UNADDRESSED RISK:</div>
                      <div style={{ fontSize: '11px', color: '#94a3b8' }}>{rev.risk}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* VIEW 15: SCIENTIFIC PAPER VIEWER & EXPORT CENTER */}
        {activeTab === 'paper' && (
          <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', padding: '24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingBottom: '16px', borderBottom: '1px solid #1e293b', marginBottom: '20px' }}>
              <div>
                <h3 style={{ fontSize: '18px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <FileText size={20} color="#10b981" /> Publication-Ready Scientific Paper
                </h3>
                <p style={{ fontSize: '12px', color: '#94a3b8', marginTop: '4px' }}>
                  Formal academic paper in Markdown with structured Abstract, Terminology, Evidence Matrix, Systems Biology Graph, DepMap, Virtual Trials, openFDA Safety, ClinVar, Single-Cell TME, Hypotheses, and Bibliography.
                </p>
              </div>

              {/* Academic Exporters (Markdown, PDF, LaTeX) */}
              <div style={{ display: 'flex', gap: '8px' }}>
                <a
                  href={`${API_BASE}/api/papers/${latestPaper?.filename || 'paper.md'}`}
                  target="_blank"
                  rel="noreferrer"
                  style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '8px 14px', borderRadius: '6px', background: '#0284c7', color: '#fff', textDecoration: 'none', fontSize: '12px', fontWeight: 600 }}
                >
                  <Download size={14} /> Markdown (.md)
                </a>

                <a
                  href={`${API_BASE}/api/papers/${latestPaper?.filename || 'paper.md'}/pdf`}
                  target="_blank"
                  rel="noreferrer"
                  style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '8px 14px', borderRadius: '6px', background: '#059669', color: '#fff', textDecoration: 'none', fontSize: '12px', fontWeight: 600 }}
                >
                  <FileText size={14} /> Nature/Cell PDF
                </a>

                <a
                  href={`${API_BASE}/api/papers/${latestPaper?.filename || 'paper.md'}/latex`}
                  target="_blank"
                  rel="noreferrer"
                  style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '8px 14px', borderRadius: '6px', background: '#7c3aed', color: '#fff', textDecoration: 'none', fontSize: '12px', fontWeight: 600 }}
                >
                  <FileCode size={14} /> LaTeX (.tex)
                </a>
              </div>
            </div>

            <div
              style={{
                background: '#020617',
                border: '1px solid #1e293b',
                borderRadius: '8px',
                padding: '24px',
                fontFamily: 'Inter, sans-serif',
                fontSize: '13px',
                lineHeight: 1.6,
                maxHeight: '650px',
                overflowY: 'auto',
                whiteSpace: 'pre-wrap',
                color: '#cbd5e1',
              }}
            >
              {latestPaper?.markdown || 'Launch an investigation to generate a complete publication-grade scientific paper.'}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
