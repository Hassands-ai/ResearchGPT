/* ============================================================
   RESEARCHGPT API — PART 2 / 2
   ============================================================ */

/* ============================================================
   GLOBAL EXPORTS
   ============================================================ */

window.API_BASE = API_BASE;

window.getToken = getToken;
window.setToken = setToken;
window.clearToken = clearToken;
window.apiRequest = apiRequest;


/* ============================================================
   AUTH
   ============================================================ */

window.login = login;
window.register = register;
window.getMe = getMe;


/* ============================================================
   PROJECTS
   ============================================================ */

window.getProjects = getProjects;
window.createProject = createProject;


/* ============================================================
   PAPERS
   ============================================================ */

window.getPapers = getPapers;
window.uploadPaper = uploadPaper;
window.processPaper = processPaper;
window.deletePaper = deletePaper;
window.searchPapers = searchPapers;


/* ============================================================
   AI CHAT
   ============================================================ */

window.chatWithPaper = chatWithPaper;
window.multiDocumentSearch = multiDocumentSearch;


/* ============================================================
   RESEARCH TOOLS
   ============================================================ */

window.comparePapers = comparePapers;
window.generateLiteratureReview =
    generateLiteratureReview;

window.generateResearchGap =
    generateResearchGap;

window.generateCitationManager =
    generateCitationManager;


/* ============================================================
   PAPER WRITE-UP
   ============================================================ */

window.generatePaperWriteup =
    generatePaperWriteup;


/* ============================================================
   PAPER ID HELPERS
   ============================================================ */

window.normalizePaperId =
    normalizePaperId;

window.normalizePaperIds =
    normalizePaperIds;


/* ============================================================
   BACKWARD COMPATIBILITY
   ============================================================ */

/*
 * Older frontend code may still reference PaperAxiom.
 * Keep these aliases so changing the visible product name
 * does not break existing functionality.
 */

window.PaperAxiomAPI = {
    apiRequest,
    getToken,
    setToken,
    clearToken,

    login,
    register,
    getMe,

    getProjects,
    createProject,

    getPapers,
    uploadPaper,
    processPaper,
    deletePaper,
    searchPapers,

    chatWithPaper,
    multiDocumentSearch,

    comparePapers,
    generateLiteratureReview,
    generateResearchGap,
    generateCitationManager,

    generatePaperWriteup
};


/* ============================================================
   API STATUS HELPERS
   ============================================================ */

async function checkBackendHealth() {

    try {

        const response =
            await fetch(
                `${API_BASE.replace(
                    "/api/v1",
                    ""
                )}/health`,
                {
                    method: "GET",
                    cache: "no-store"
                }
            );

        if (!response.ok) {
            return false;
        }

        const data =
            await response.json();

        return data?.status === "healthy";

    } catch {

        return false;
    }
}


window.checkBackendHealth =
    checkBackendHealth;


/* ============================================================
   CACHE HELPERS
   ============================================================ */

function clearPaperCache() {

    /*
     * app.js maintains its own in-memory cache.
     * This event allows the frontend to invalidate it
     * after upload/delete/process operations.
     */

    window.dispatchEvent(
        new CustomEvent(
            "researchGPT:papersChanged"
        )
    );
}


window.clearPaperCache =
    clearPaperCache;


/* ============================================================
   SAFE PAPER LIST
   ============================================================ */

async function getPapersSafe(
    timeout = 8000
) {

    try {

        return await Promise.race([

            getPapers(),

            new Promise(
                (_, reject) =>
                    setTimeout(
                        () =>
                            reject(
                                new Error(
                                    "Paper loading timed out."
                                )
                            ),
                        timeout
                    )
            )

        ]);

    } catch (error) {

        console.warn(
            "ResearchGPT: paper loading failed:",
            error
        );

        return [];
    }
}


window.getPapersSafe =
    getPapersSafe;


/* ============================================================
   VALIDATION
   ============================================================ */

function validatePaperIds(
    paperIds
) {

    if (
        !Array.isArray(
            paperIds
        )
    ) {

        throw new Error(
            "Paper IDs must be provided as a list."
        );
    }


    const normalized =
        normalizePaperIds(
            paperIds
        );


    if (
        normalized.length === 0
    ) {

        throw new Error(
            "Please select at least one paper."
        );
    }


    if (
        normalized.length > 10
    ) {

        throw new Error(
            "You can select a maximum of 10 papers."
        );
    }


    return normalized;
}


window.validatePaperIds =
    validatePaperIds;


/* ============================================================
   IMPROVED COMPARISON
   ============================================================ */

async function compareSelectedPapers(
    paperIds
) {

    const ids =
        validatePaperIds(
            paperIds
        );


    if (
        ids.length < 2
    ) {

        throw new Error(
            "Please select at least 2 papers for comparison."
        );
    }


    return comparePapers(
        ids,
        6
    );
}


window.compareSelectedPapers =
    compareSelectedPapers;


/* ============================================================
   IMPROVED RESEARCH GAP
   ============================================================ */

async function generateSelectedResearchGap(
    paperIds
) {

    const ids =
        validatePaperIds(
            paperIds
        );


    return generateResearchGap(
        ids
    );
}


window.generateSelectedResearchGap =
    generateSelectedResearchGap;


/* ============================================================
   IMPROVED CITATION MANAGER
   ============================================================ */

async function generateSelectedCitationManager(
    paperIds
) {

    const ids =
        validatePaperIds(
            paperIds
        );


    return generateCitationManager(
        ids
    );
}


window.generateSelectedCitationManager =
    generateSelectedCitationManager;


/* ============================================================
   IMPROVED LITERATURE REVIEW
   ============================================================ */

async function generateSelectedLiteratureReview(
    paperIds
) {

    const ids =
        validatePaperIds(
            paperIds
        );


    return generateLiteratureReview(
        ids
    );
}


window.generateSelectedLiteratureReview =
    generateSelectedLiteratureReview;


/* ============================================================
   IMPROVED WRITE-UP
   ============================================================ */

async function generateSelectedPaperWriteup(
    paperIds,
    section = "introduction",
    topic = "",
    instructions = ""
) {

    const ids =
        validatePaperIds(
            paperIds
        );


    return generatePaperWriteup(
        ids,
        section,
        topic,
        instructions
    );
}


window.generateSelectedPaperWriteup =
    generateSelectedPaperWriteup;


/* ============================================================
   UPLOAD WRAPPER
   ============================================================ */

async function uploadAndProcessPaper(
    file,
    title = "",
    projectId = null
) {

    if (!file) {

        throw new Error(
            "Please select a PDF file."
        );
    }


    if (
        !file.name
            .toLowerCase()
            .endsWith(".pdf")
    ) {

        throw new Error(
            "Only PDF files are supported."
        );
    }


    const uploaded =
        await uploadPaper(
            file,
            title ||
                file.name.replace(
                    /\.pdf$/i,
                    ""
                ),
            projectId
        );


    /*
     * Process only when the backend returned
     * a valid paper ID.
     */

    const paperId =
        normalizePaperId(
            uploaded
        );


    if (
        paperId !== null &&
        typeof processPaper ===
            "function"
    ) {

        try {

            await processPaper(
                paperId
            );

        } catch (error) {

            /*
             * Upload succeeded even if processing
             * fails. Do not hide the uploaded paper.
             */

            console.warn(
                "ResearchGPT: paper processing failed:",
                error
            );
        }
    }


    clearPaperCache();


    return uploaded;
}


window.uploadAndProcessPaper =
    uploadAndProcessPaper;


/* ============================================================
   RESPONSE NORMALIZATION
   ============================================================ */

function extractResponseText(
    response
) {

    if (
        response === null ||
        response === undefined
    ) {
        return "";
    }


    if (
        typeof response === "string"
    ) {
        return response;
    }


    const candidates = [

        response.answer,
        response.response,
        response.result,
        response.content,
        response.text,
        response.analysis,
        response.summary,
        response.writeup,
        response.paper_writeup,
        response.research_gap,
        response.comparison,
        response.literature_review

    ];


    for (
        const candidate
        of candidates
    ) {

        if (
            typeof candidate ===
                "string" &&
            candidate.trim()
        ) {

            return candidate;
        }
    }


    return JSON.stringify(
        response,
        null,
        2
    );
}


window.extractResponseText =
    extractResponseText;


/* ============================================================
   ERROR NORMALIZATION
   ============================================================ */

function getApiErrorMessage(
    error
) {

    if (!error) {
        return "An unknown error occurred.";
    }


    if (
        typeof error ===
        "string"
    ) {
        return error;
    }


    return (
        error.message ||
        error.detail ||
        "An unexpected API error occurred."
    );
}


window.getApiErrorMessage =
    getApiErrorMessage;


/* ============================================================
   API READY EVENT
   ============================================================ */

window.dispatchEvent(
    new CustomEvent(
        "researchGPT:apiReady"
    )
);


/* ============================================================
   STARTUP MESSAGE
   ============================================================ */

console.log(
    "ResearchGPT API loaded successfully."
);
console.log(
    `API Base: ${API_BASE}`
);
console.log(
    "Available research tools: Chat, Literature Review, Research Gap, Paper Comparison, Citation Manager, Paper Write-up"
);