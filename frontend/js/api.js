"use strict";

/* ============================================================
   PaperAxiom API Client
   Frontend <-> FastAPI bridge
   ============================================================ */

const API_BASE = "/api/v1";


/* ============================================================
   AUTH TOKEN
   ============================================================ */

function getToken() {
    return (
        localStorage.getItem("token") ||
        localStorage.getItem("access_token") ||
        sessionStorage.getItem("access_token") ||
        ""
    );
}


function setToken(token) {
    if (!token) {
        return;
    }

    localStorage.setItem("token", token);
    localStorage.setItem("access_token", token);
}


function clearToken() {
    localStorage.removeItem("token");
    localStorage.removeItem("access_token");
    sessionStorage.removeItem("access_token");
}


/* ============================================================
   API REQUEST
   ============================================================ */

async function apiRequest(endpoint, options = {}) {

    const token = getToken();

    const headers = {
        ...(options.headers || {})
    };


    /* --------------------------------------------------------
       Authorization
       -------------------------------------------------------- */

    if (token) {
        headers["Authorization"] = `Bearer ${token}`;
    }


    /* --------------------------------------------------------
       IMPORTANT:
       Do NOT manually set Content-Type for FormData.
       Browser automatically creates multipart boundary.
       -------------------------------------------------------- */

    if (!(options.body instanceof FormData)) {

        headers["Content-Type"] =
            "application/json";
    }


    let response;


    /* --------------------------------------------------------
       Send request
       -------------------------------------------------------- */

    try {

        response = await fetch(
            `${API_BASE}${endpoint}`,
            {
                ...options,
                headers
            }
        );

    } catch (error) {

        console.error(
            "PaperAxiom API connection error:",
            error
        );

        throw new Error(
            "Unable to connect to PaperAxiom backend. " +
            "Make sure FastAPI is running on port 8000."
        );
    }


    /* --------------------------------------------------------
       Authentication error
       -------------------------------------------------------- */

    if (response.status === 401) {

        clearToken();

        if (
            !window.location.pathname.endsWith(
                "login.html"
            )
        ) {

            window.location.href =
                "login.html";
        }

        throw new Error(
            "Your session has expired."
        );
    }


    /* --------------------------------------------------------
       Parse response safely
       -------------------------------------------------------- */

    let data = null;

    const contentType =
        response.headers.get(
            "content-type"
        ) || "";


    try {

        if (
            contentType.includes(
                "application/json"
            )
        ) {

            data =
                await response.json();

        } else {

            const text =
                await response.text();

            data =
                text
                    ? {
                        detail: text
                    }
                    : null;
        }

    } catch (error) {

        data = null;
    }


    /* --------------------------------------------------------
       Handle API errors
       -------------------------------------------------------- */

    if (!response.ok) {

        let message =
            `Request failed (${response.status}).`;


        if (data?.detail) {

            message =
                typeof data.detail === "string"
                    ? data.detail
                    : JSON.stringify(
                        data.detail
                    );

        } else if (data?.message) {

            message =
                data.message;

        } else if (data?.error) {

            message =
                data.error;
        }


        throw new Error(message);
    }


    return data;
}


/* ============================================================
   AUTH
   ============================================================ */


/* -------------------- LOGIN -------------------- */

async function login(
    email,
    password
) {

    const form =
        new URLSearchParams();


    form.append(
        "username",
        email
    );

    form.append(
        "password",
        password
    );


    const response =
        await fetch(
            `${API_BASE}/auth/login`,
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/x-www-form-urlencoded"
                },

                body: form
            }
        );


    let data = null;


    try {

        data =
            await response.json();

    } catch (error) {

        data = null;
    }


    if (!response.ok) {

        throw new Error(
            data?.detail ||
            data?.message ||
            "Login failed."
        );
    }


    setToken(
        data.access_token
    );


    return data;
}


/* -------------------- REGISTER -------------------- */

async function register(
    email,
    fullName,
    password
) {

    return apiRequest(
        "/auth/register",
        {
            method: "POST",

            body: JSON.stringify({
                email: email,
                full_name: fullName,
                password: password
            })
        }
    );
}


/* -------------------- CURRENT USER -------------------- */

async function getMe() {

    return apiRequest(
        "/auth/me"
    );
}


/* ============================================================
   PROJECTS
   ============================================================ */


/* -------------------- GET PROJECTS -------------------- */

async function getProjects() {

    const result =
        await apiRequest(
            "/projects"
        );


    if (
        Array.isArray(result)
    ) {

        return result;
    }


    return (
        result?.projects ||
        result?.items ||
        result?.data ||
        []
    );
}


/* -------------------- CREATE PROJECT -------------------- */

async function createProject(
    title,
    description = ""
) {

    return apiRequest(
        "/projects",
        {
            method: "POST",

            body: JSON.stringify({
                title: title,
                description: description
            })
        }
    );
}


/* ============================================================
   PAPERS
   ============================================================ */


/* -------------------- GET PAPERS -------------------- */

async function getPapers() {

    const result =
        await apiRequest(
            "/papers"
        );


    if (
        Array.isArray(result)
    ) {

        return result;
    }


    return (
        result?.papers ||
        result?.items ||
        result?.data ||
        []
    );
}


/* ============================================================
   UPLOAD PAPER
   POST /papers/upload
   ============================================================ */

async function uploadPaper(
    file,
    title = "",
    projectId = null
) {

    /* --------------------------------------------------------
       Validate file
       -------------------------------------------------------- */

    if (!file) {

        throw new Error(
            "Please select a PDF file."
        );
    }


    /* --------------------------------------------------------
       Validate PDF
       -------------------------------------------------------- */

    if (
        file.type &&
        file.type !== "application/pdf" &&
        !file.name
            .toLowerCase()
            .endsWith(".pdf")
    ) {

        throw new Error(
            "Only PDF files are allowed."
        );
    }


    /* --------------------------------------------------------
       Create multipart form
       -------------------------------------------------------- */

    const form =
        new FormData();


    form.append(
        "file",
        file
    );


    form.append(
        "title",
        title ||
        file.name.replace(
            /\.pdf$/i,
            ""
        )
    );


    /* --------------------------------------------------------
       Optional project ID
       -------------------------------------------------------- */

    if (
        projectId !== null &&
        projectId !== undefined &&
        projectId !== ""
    ) {

        form.append(
            "project_id",
            projectId
        );
    }


    /* --------------------------------------------------------
       Send upload request
       -------------------------------------------------------- */

    return apiRequest(
        "/papers/upload",
        {
            method: "POST",
            body: form
        }
    );
}


/* ============================================================
   PROCESS PAPER
   POST /papers/{paper_id}/process
   ============================================================ */

async function processPaper(
    paperId
) {

    if (
        paperId === null ||
        paperId === undefined ||
        paperId === ""
    ) {

        throw new Error(
            "Invalid paper ID."
        );
    }


    return apiRequest(
        `/papers/${Number(paperId)}/process`,
        {
            method: "POST"
        }
    );
}


/* ============================================================
   DELETE PAPER
   DELETE /papers/{paper_id}
   ============================================================ */

async function deletePaper(
    paperId
) {

    if (
        paperId === null ||
        paperId === undefined ||
        paperId === ""
    ) {

        throw new Error(
            "Invalid paper ID."
        );
    }


    return apiRequest(
        `/papers/${Number(paperId)}`,
        {
            method: "DELETE"
        }
    );
}


/* ============================================================
   PAPER SEARCH
   POST /papers/search
   ============================================================ */

async function searchPapers(
    query,
    paperId = null,
    limit = 5
) {

    return apiRequest(
        "/papers/search",
        {
            method: "POST",

            body: JSON.stringify({

                query:
                    query || "",

                paper_id:
                    paperId === null ||
                    paperId === undefined ||
                    paperId === ""
                        ? null
                        : Number(paperId),

                limit:
                    Number(limit) || 5
            })
        }
    );
}


/* ============================================================
   AI CHAT
   POST /papers/chat
   ============================================================ */

async function chatWithPaper(
    question,
    paperId,
    limit = 5
) {

    if (
        !question ||
        !question.trim()
    ) {

        throw new Error(
            "Question cannot be empty."
        );
    }


    if (
        paperId === null ||
        paperId === undefined ||
        paperId === ""
    ) {

        throw new Error(
            "Please select a paper."
        );
    }


    return apiRequest(
        "/papers/chat",
        {
            method: "POST",

            body: JSON.stringify({

                question:
                    question.trim(),

                paper_id:
                    Number(paperId),

                limit:
                    Number(limit) || 5
            })
        }
    );
}


/* ============================================================
   MULTI-DOCUMENT SEARCH
   POST /papers/multi-search
   ============================================================ */

async function multiDocumentSearch(
    query,
    paperIds,
    limitPerPaper = 5
) {

    validatePaperIds(
        paperIds
    );


    return apiRequest(
        "/papers/multi-search",
        {
            method: "POST",

            body: JSON.stringify({

                query:
                    query || "",

                paper_ids:
                    paperIds.map(Number),

                limit_per_paper:
                    Number(
                        limitPerPaper
                    ) || 5
            })
        }
    );
}


/* ============================================================
   PAPER COMPARISON
   POST /papers/compare
   ============================================================ */

async function comparePapers(
    paperIds,
    evidencePerPaper = 5,
    userQuestion = ""
) {

    if (
        !Array.isArray(paperIds) ||
        paperIds.length < 2
    ) {

        throw new Error(
            "Please select at least two papers for comparison."
        );
    }


    if (
        paperIds.length > 10
    ) {

        throw new Error(
            "You can select a maximum of 10 papers."
        );
    }


    const body = {

        paper_ids:
            paperIds.map(Number),

        evidence_per_paper:
            Number(
                evidencePerPaper
            ) || 5
    };


    if (
        userQuestion &&
        userQuestion.trim()
    ) {

        body.user_question =
            userQuestion.trim();
    }


    return apiRequest(
        "/papers/compare",
        {
            method: "POST",

            body:
                JSON.stringify(
                    body
                )
        }
    );
}


/* ============================================================
   LITERATURE REVIEW
   POST /papers/literature-review
   ============================================================ */

async function generateLiteratureReview(
    paperIds
) {

    validatePaperIds(
        paperIds
    );


    return apiRequest(
        "/papers/literature-review",
        {
            method: "POST",

            body: JSON.stringify({

                paper_ids:
                    paperIds.map(Number)
            })
        }
    );
}


/* ============================================================
   RESEARCH GAP
   POST /papers/research-gap
   ============================================================ */

async function generateResearchGap(
    paperIds
) {

    validatePaperIds(
        paperIds
    );


    return apiRequest(
        "/papers/research-gap",
        {
            method: "POST",

            body: JSON.stringify({

                paper_ids:
                    paperIds.map(Number)
            })
        }
    );
}


/* ============================================================
   CITATION MANAGER
   POST /papers/citation-manager
   ============================================================ */

async function generateCitationManager(
    paperIds
) {

    validatePaperIds(
        paperIds
    );


    return apiRequest(
        "/papers/citation-manager",
        {
            method: "POST",

            body: JSON.stringify({

                paper_ids:
                    paperIds.map(Number)
            })
        }
    );
}


/* ============================================================
   PAPER WRITE-UP
   POST /papers/writeup
   ============================================================ */

async function generatePaperWriteup(
    paperIds,
    writeupType = "introduction",
    researchTopic = "",
    instructions = ""
) {

    validatePaperIds(
        paperIds
    );


    const normalizedPaperIds =
        paperIds
            .map(Number)
            .filter(
                id =>
                    Number.isInteger(id) &&
                    id > 0
            );


    if (
        normalizedPaperIds.length === 0
    ) {

        throw new Error(
            "No valid paper IDs were selected."
        );
    }


    return apiRequest(
        "/papers/writeup",
        {
            method: "POST",

            body: JSON.stringify({

                paper_ids:
                    normalizedPaperIds,

                writeup_type:
                    writeupType ||
                    "introduction",

                research_topic:
                    researchTopic ||
                    null,

                instructions:
                    instructions ||
                    null
            })
        }
    );
}


/* ============================================================
   VALIDATE PAPER IDS
   ============================================================ */

function validatePaperIds(
    paperIds
) {

    if (
        !Array.isArray(paperIds) ||
        paperIds.length === 0
    ) {

        throw new Error(
            "Please select at least one paper."
        );
    }


    if (
        paperIds.length > 10
    ) {

        throw new Error(
            "You can select a maximum of 10 papers."
        );
    }


    const validIds =
        paperIds
            .map(Number)
            .filter(
                id =>
                    Number.isInteger(id) &&
                    id > 0
            );


    if (
        validIds.length === 0
    ) {

        throw new Error(
            "No valid paper IDs were selected."
        );
    }
}


/* ============================================================
   PAPER ID HELPERS
   ============================================================ */

function normalizePaperId(
    paper
) {

    if (
        paper === null ||
        paper === undefined
    ) {

        return null;
    }


    if (
        typeof paper === "object"
    ) {

        paper =
            paper.id ??
            paper.paper_id ??
            paper.paperId;
    }


    const value =
        Number(paper);


    return Number.isFinite(
        value
    )
        ? value
        : null;
}


function normalizePaperIds(
    papers
) {

    if (
        !Array.isArray(papers)
    ) {

        return [];
    }


    return [
        ...new Set(
            papers
                .map(
                    normalizePaperId
                )
                .filter(
                    id =>
                        id !== null &&
                        id > 0
                )
        )
    ];
}


/* ============================================================
   GLOBAL EXPORTS
   ============================================================

   IMPORTANT:
   app.js uses window.functionName(...)
   Therefore every API function used by app.js MUST
   be exported here.
   ============================================================ */


/* -------------------- CORE -------------------- */

window.API_BASE =
    API_BASE;

window.getToken =
    getToken;

window.setToken =
    setToken;

window.clearToken =
    clearToken;

window.apiRequest =
    apiRequest;


/* -------------------- AUTH -------------------- */

window.login =
    login;

window.register =
    register;

window.getMe =
    getMe;


/* -------------------- PROJECTS -------------------- */

window.getProjects =
    getProjects;

window.createProject =
    createProject;


/* -------------------- PAPERS -------------------- */

window.getPapers =
    getPapers;

window.uploadPaper =
    uploadPaper;

window.processPaper =
    processPaper;

window.deletePaper =
    deletePaper;

window.searchPapers =
    searchPapers;


/* -------------------- AI -------------------- */

window.chatWithPaper =
    chatWithPaper;

window.multiDocumentSearch =
    multiDocumentSearch;

window.comparePapers =
    comparePapers;

window.generateLiteratureReview =
    generateLiteratureReview;

window.generateResearchGap =
    generateResearchGap;

window.generateCitationManager =
    generateCitationManager;

window.generatePaperWriteup =
    generatePaperWriteup;


/* -------------------- HELPERS -------------------- */

window.validatePaperIds =
    validatePaperIds;

window.normalizePaperId =
    normalizePaperId;

window.normalizePaperIds =
    normalizePaperIds;


/* ============================================================
   DEBUG
   ============================================================ */

console.log(
    "PaperAxiom API loaded successfully."
);

console.log(
    "API Base:",
    API_BASE
);

console.log(
    "uploadPaper available:",
    typeof window.uploadPaper === "function"
);

console.log(
    "processPaper available:",
    typeof window.processPaper === "function"
);

console.log(
    "getPapers available:",
    typeof window.getPapers === "function"
);