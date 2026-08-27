"use strict";

/* ============================================================
   ResearchGPT
   File: frontend/js/app.js

   FAST / STABLE FRONTEND CONTROLLER

   Main rule:
   The dashboard must render immediately.

   API calls are background operations.
   They must NEVER block the initial UI.
   ============================================================ */


/* ============================================================
   GLOBAL STATE
   ============================================================ */

let currentUser = null;
let currentPage = "dashboard";

let papersCache = [];
let projectsCache = [];

let navigationRequestId = 0;

const MAX_PAPERS = 10;
const API_TIMEOUT = 10000;


/* ============================================================
   PAGE CONFIGURATION
   ============================================================ */

const PAGE_TITLES = {

    dashboard: [
        "Dashboard",
        "Your AI research workspace"
    ],

    projects: [
        "My Projects",
        "Organize your research work"
    ],

    upload: [
        "Library & Papers",
        "Upload and manage research papers"
    ],

    chat: [
        "AI Chat",
        "Ask evidence-grounded questions about your papers"
    ],

    literature: [
        "Literature Review",
        "Generate an evidence-grounded academic synthesis"
    ],

    "research-gap": [
        "Research Gap",
        "Identify limitations, unresolved problems and opportunities"
    ],

    comparison: [
        "Paper Comparison",
        "Compare 2–10 research papers"
    ],

    citation: [
        "Citation Manager",
        "Generate academic citation information"
    ],

    writeup: [
        "Paper Write-up",
        "Generate focused academic content from selected papers"
    ],

    settings: [
        "Settings",
        "ResearchGPT workspace settings"
    ]
};


/* ============================================================
   BASIC HELPERS
   ============================================================ */

function escapeHtml(value) {

    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}


function getPaperId(paper) {

    if (
        paper === null ||
        paper === undefined
    ) {
        return null;
    }

    if (
        typeof paper === "number"
    ) {
        return Number.isFinite(paper)
            ? paper
            : null;
    }

    if (
        typeof paper === "string"
    ) {

        const value = Number(paper);

        return Number.isFinite(value)
            ? value
            : null;
    }

    const value =
        paper.id ??
        paper.paper_id ??
        paper.paperId ??
        null;

    const number =
        Number(value);

    return Number.isFinite(number)
        ? number
        : null;
}


function getPaperTitle(paper) {

    if (!paper) {
        return "Research Paper";
    }

    return (
        paper.title ||
        paper.name ||
        paper.paper_title ||
        paper.filename ||
        paper.file_name ||
        `Research Paper ${getPaperId(paper) ?? ""}`
    );
}


function getPaperAuthors(paper) {

    if (!paper) {
        return "";
    }

    const authors =
        paper.authors ||
        paper.author ||
        "";

    if (
        Array.isArray(authors)
    ) {
        return authors.join(", ");
    }

    return String(authors);
}


/* ============================================================
   SAFE API TIMEOUT
   ============================================================ */

function timeoutPromise(
    promise,
    timeout = API_TIMEOUT,
    message = "Request timed out."
) {

    let timer = null;

    const timeoutRequest =
        new Promise(
            (_, reject) => {

                timer =
                    setTimeout(
                        () => {
                            reject(
                                new Error(
                                    message
                                )
                            );
                        },
                        timeout
                    );
            }
        );

    return Promise
        .race([
            promise,
            timeoutRequest
        ])
        .finally(
            () => {
                if (timer) {
                    clearTimeout(timer);
                }
            }
        );
}


/* ============================================================
   TOAST
   ============================================================ */

function showToast(
    message,
    type = "info"
) {

    let toast =
        document.getElementById(
            "researchGPTToast"
        );

    if (!toast) {

        toast =
            document.createElement(
                "div"
            );

        toast.id =
            "researchGPTToast";

        document.body.appendChild(
            toast
        );
    }

    let background =
        "bg-slate-800";

    if (
        type === "success"
    ) {
        background =
            "bg-emerald-600";
    }

    if (
        type === "error"
    ) {
        background =
            "bg-red-600";
    }

    toast.className = `
        fixed
        bottom-5
        right-5
        z-[9999]
        max-w-sm
        px-4
        py-3
        rounded-xl
        shadow-xl
        text-sm
        font-medium
        text-white
        ${background}
    `;

    toast.textContent =
        message;

    clearTimeout(
        window.__researchGPTToastTimer
    );

    window.__researchGPTToastTimer =
        setTimeout(
            () => {
                toast.remove();
            },
            3500
        );
}


/* ============================================================
   LOADING UI
   ============================================================ */

function setLoading(
    container,
    message = "Loading..."
) {

    if (!container) {
        return;
    }

    container.innerHTML = `
        <div class="
            min-h-[260px]
            flex
            flex-col
            items-center
            justify-center
            text-center
        ">

            <div class="
                w-9
                h-9
                rounded-full
                border-4
                border-indigo-100
                border-t-indigo-600
                animate-spin
            "></div>

            <p class="
                mt-4
                text-sm
                text-slate-500
                dark:text-slate-400
            ">
                ${escapeHtml(message)}
            </p>

        </div>
    `;
}


function emptyState(
    icon,
    title,
    description
) {

    return `
        <div class="
            min-h-[230px]
            flex
            flex-col
            items-center
            justify-center
            text-center
            px-6
        ">

            <div class="
                text-4xl
                mb-4
            ">
                ${icon}
            </div>

            <h3 class="
                text-lg
                font-semibold
                text-slate-800
                dark:text-white
            ">
                ${escapeHtml(title)}
            </h3>

            <p class="
                max-w-lg
                text-sm
                text-slate-500
                dark:text-slate-400
                mt-2
                leading-6
            ">
                ${escapeHtml(description)}
            </p>

        </div>
    `;
}


/* ============================================================
   GENERATED TEXT
   ============================================================ */

function formatGeneratedText(
    text
) {

    if (
        !text
    ) {
        return `
            <p class="text-slate-400">
                No response returned.
            </p>
        `;
    }

    let value =
        escapeHtml(text);

    value =
        value.replace(
            /^### (.*)$/gm,
            "<h3 class='text-lg font-bold mt-6 mb-2'>$1</h3>"
        );

    value =
        value.replace(
            /^## (.*)$/gm,
            "<h2 class='text-xl font-bold mt-7 mb-3'>$1</h2>"
        );

    value =
        value.replace(
            /^# (.*)$/gm,
            "<h1 class='text-2xl font-bold mt-7 mb-3'>$1</h1>"
        );

    value =
        value.replace(
            /\*\*(.*?)\*\*/g,
            "<strong>$1</strong>"
        );

    value =
        value.replace(
            /^[-•] (.*)$/gm,
            "<li class='ml-5 list-disc mb-1'>$1</li>"
        );

    value =
        value.replace(
            /\n\n/g,
            "</p><p class='mb-4'>"
        );

    value =
        value.replace(
            /\n/g,
            "<br>"
        );

    return `
        <div class="
            prose
            prose-slate
            dark:prose-invert
            max-w-none
            leading-7
        ">
            <p class="mb-4">
                ${value}
            </p>
        </div>
    `;
}


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
            typeof candidate === "string" &&
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


/* ============================================================
   FAST BACKGROUND DATA LOADING
   ============================================================ */

async function loadPapersBackground() {

    if (
        typeof window.getPapers !==
        "function"
    ) {
        return [];
    }

    try {

        const result =
            await timeoutPromise(
                window.getPapers(),
                API_TIMEOUT,
                "Paper loading timed out."
            );

        papersCache =
            Array.isArray(result)
                ? result
                : (
                    result?.papers ||
                    result?.items ||
                    result?.data ||
                    []
                );

        return papersCache;

    } catch (error) {

        console.warn(
            "ResearchGPT paper loading:",
            error.message
        );

        return papersCache;
    }
}


async function loadProjectsBackground() {

    if (
        typeof window.getProjects !==
        "function"
    ) {
        return [];
    }

    try {

        const result =
            await timeoutPromise(
                window.getProjects(),
                API_TIMEOUT,
                "Project loading timed out."
            );

        projectsCache =
            Array.isArray(result)
                ? result
                : (
                    result?.projects ||
                    result?.items ||
                    result?.data ||
                    []
                );

        return projectsCache;

    } catch (error) {

        console.warn(
            "ResearchGPT project loading:",
            error.message
        );

        return projectsCache;
    }
}


/* ============================================================
   USER — BACKGROUND ONLY
   ============================================================ */

async function loadCurrentUserBackground() {

    if (
        typeof window.getMe !==
        "function"
    ) {
        return null;
    }

    try {

        const user =
            await timeoutPromise(
                window.getMe(),
                7000,
                "User loading timed out."
            );

        currentUser =
            user || null;

        updateUserHeader();

        return currentUser;

    } catch (error) {

        console.warn(
            "ResearchGPT user loading:",
            error.message
        );

        /*
         * DO NOT redirect here.
         *
         * This is important.
         * A slow backend must not make the whole
         * frontend appear frozen.
         */

        return null;
    }
}


function updateUserHeader() {

    const name =
        currentUser?.full_name ||
        currentUser?.name ||
        "Researcher";

    const avatar =
        document.getElementById(
            "userAvatar"
        );

    if (avatar) {

        avatar.textContent =
            name
                .trim()
                .charAt(0)
                .toUpperCase() ||
            "R";
    }

    const userName =
        document.getElementById(
            "userName"
        );

    if (userName) {
        userName.textContent =
            name;
    }

    const userEmail =
        document.getElementById(
            "userEmail"
        );

    if (userEmail) {

        userEmail.textContent =
            currentUser?.email ||
            "";
    }
}


/* ============================================================
   NAVIGATION
   ============================================================ */

function updateNavigation(
    page
) {

    document
        .querySelectorAll(
            ".sidebar-item[data-page]"
        )
        .forEach(
            item => {

                item.classList.toggle(
                    "active",
                    item.dataset.page ===
                    page
                );
            }
        );
}


function updatePageHeader(
    page
) {

    const title =
        document.querySelector(
            "[data-page-title]"
        ) ||
        document.getElementById(
            "pageTitle"
        );

    const subtitle =
        document.querySelector(
            "[data-page-subtitle]"
        ) ||
        document.getElementById(
            "pageSubtitle"
        );

    const values =
        PAGE_TITLES[page] ||
        PAGE_TITLES.dashboard;

    if (title) {
        title.textContent =
            values[0];
    }

    if (subtitle) {
        subtitle.textContent =
            values[1];
    }
}


/* ============================================================
   MAIN ROUTER
   ============================================================ */

async function loadPage(
    page
) {

    if (
        !PAGE_TITLES[page]
    ) {
        page =
            "dashboard";
    }

    currentPage =
        page;

    const requestId =
        ++navigationRequestId;

    updateNavigation(
        page
    );

    updatePageHeader(
        page
    );

    const container =
        document.querySelector(
            "[data-page-content]"
        ) ||
        document.getElementById(
            "pageContent"
        ) ||
        document.getElementById(
            "app"
        ) ||
        document.querySelector(
            "main"
        );

    if (!container) {
        return;
    }


    /*
     * DASHBOARD SPECIAL CASE
     *
     * Render immediately.
     * Never wait for API.
     */

    if (
        page === "dashboard"
    ) {

        renderDashboard(
            container
        );

        /*
         * Background refresh only.
         */

        Promise.allSettled([
            loadPapersBackground(),
            loadProjectsBackground()
        ])
            .then(
                () => {

                    if (
                        currentPage ===
                        "dashboard"
                    ) {

                        updateDashboardData(
                            container
                        );
                    }
                }
            );

        return;
    }


    /*
     * All secondary pages.
     */

    setLoading(
        container,
        `Loading ${PAGE_TITLES[page][0]}...`
    );


    try {

        switch (page) {

            case "projects":

                await renderProjects(
                    container
                );

                break;


            case "upload":

                await renderLibrary(
                    container
                );

                break;


            case "chat":

                await renderChat(
                    container
                );

                break;


            case "literature":

                await renderLiterature(
                    container
                );

                break;


            case "research-gap":

                await renderResearchGap(
                    container
                );

                break;


            case "comparison":

                await renderComparison(
                    container
                );

                break;


            case "citation":

                await renderCitationManager(
                    container
                );

                break;


            case "writeup":

                await renderPaperWriteup(
                    container
                );

                break;


            case "settings":

                await renderSettings(
                    container
                );

                break;
        }

    } catch (error) {

        if (
            requestId !==
            navigationRequestId
        ) {
            return;
        }

        console.error(
            "ResearchGPT page error:",
            error
        );

        container.innerHTML = `
            <div class="
                max-w-4xl
                mx-auto
                p-6
            ">

                <div class="
                    rounded-2xl
                    border
                    border-red-200
                    bg-red-50
                    dark:bg-red-950/30
                    p-6
                ">

                    <h3 class="
                        font-semibold
                        text-red-700
                        dark:text-red-400
                    ">
                        ${escapeHtml(
                            PAGE_TITLES[page][0]
                        )}
                        failed to load
                    </h3>

                    <p class="
                        text-sm
                        text-red-600
                        dark:text-red-300
                        mt-2
                    ">
                        ${escapeHtml(
                            error.message
                        )}
                    </p>

                </div>

            </div>
        `;
    }
}


/* ============================================================
   DASHBOARD
   ============================================================ */

function renderDashboard(
    container
) {

    const firstName =
        currentUser?.full_name
            ?.trim()
            ?.split(/\s+/)[0] ||
        "Researcher";


    const papers =
        papersCache || [];

    const projects =
        projectsCache || [];


    container.innerHTML = `

        <div class="
            max-w-[1400px]
            mx-auto
            space-y-5
            pb-8
        ">


            <!-- WELCOME -->

            <section class="
                bg-white
                dark:bg-slate-800
                border
                border-slate-200
                dark:border-slate-700
                rounded-2xl
                p-6
                shadow-sm
            ">

                <div class="
                    flex
                    items-start
                    justify-between
                    gap-5
                ">

                    <div>

                        <h2 class="
                            text-2xl
                            font-bold
                            text-slate-900
                            dark:text-white
                        ">
                            Welcome back,
                            ${escapeHtml(firstName)}! 👋
                        </h2>

                        <p class="
                            text-sm
                            text-slate-500
                            dark:text-slate-400
                            mt-1
                        ">
                            Let’s accelerate your research with AI
                        </p>

                    </div>


                    <div class="
                        hidden
                        sm:flex
                        w-14
                        h-14
                        rounded-2xl
                        bg-indigo-50
                        dark:bg-indigo-900/30
                        items-center
                        justify-center
                        text-2xl
                    ">
                        🤖
                    </div>

                </div>


                <div class="
                    mt-5
                    rounded-2xl
                    bg-gradient-to-r
                    from-indigo-600
                    via-violet-600
                    to-purple-600
                    p-6
                    text-white
                ">

                    <div class="
                        text-[10px]
                        font-bold
                        uppercase
                        tracking-widest
                        text-indigo-100
                    ">
                        ResearchGPT AI Workspace
                    </div>

                    <div class="
                        text-xl
                        font-bold
                        mt-1
                    ">
                        Research faster. Understand deeper.
                    </div>

                    <p class="
                        text-sm
                        text-indigo-100
                        mt-2
                        leading-6
                        max-w-4xl
                    ">
                        Upload research papers, retrieve evidence,
                        ask grounded questions, synthesize literature,
                        identify research gaps, compare studies,
                        manage citations and prepare academic writing.
                    </p>

                </div>

            </section>


            <!-- METRICS -->

            <div class="
                grid
                grid-cols-2
                md:grid-cols-3
                lg:grid-cols-5
                gap-3
            ">

                ${dashboardMetric(
                    "📁",
                    projects.length,
                    "Active Projects"
                )}

                ${dashboardMetric(
                    "📄",
                    papers.length,
                    "Papers Uploaded"
                )}

                ${dashboardMetric(
                    "📚",
                    "—",
                    "Literature Reviews"
                )}

                ${dashboardMetric(
                    "💬",
                    "—",
                    "AI Chats"
                )}

                ${dashboardMetric(
                    "🔖",
                    "—",
                    "Saved Searches"
                )}

            </div>


            <!-- RESEARCH TOOLS -->

            <section class="
                bg-white
                dark:bg-slate-800
                border
                border-slate-200
                dark:border-slate-700
                rounded-2xl
                p-5
                shadow-sm
            ">

                <div>

                    <h3 class="
                        font-semibold
                        text-slate-900
                        dark:text-white
                    ">
                        Research Tools
                    </h3>

                    <p class="
                        text-[11px]
                        text-slate-400
                        mt-1
                    ">
                        Your core evidence-grounded research workflow
                    </p>

                </div>


                <div class="
                    grid
                    grid-cols-1
                    sm:grid-cols-2
                    lg:grid-cols-3
                    gap-3
                    mt-4
                ">

                    ${dashboardTool(
                        "📄",
                        "Paper Library",
                        "Upload and manage research papers.",
                        "upload"
                    )}

                    ${dashboardTool(
                        "💬",
                        "AI Chat",
                        "Ask evidence-grounded questions.",
                        "chat"
                    )}

                    ${dashboardTool(
                        "📚",
                        "Literature Review",
                        "Synthesize evidence across studies.",
                        "literature"
                    )}

                    ${dashboardTool(
                        "🔍",
                        "Research Gap",
                        "Find evidence-supported opportunities.",
                        "research-gap"
                    )}

                    ${dashboardTool(
                        "⚖️",
                        "Paper Comparison",
                        "Compare 2–10 research papers.",
                        "comparison"
                    )}

                    ${dashboardTool(
                        "📑",
                        "Citation Manager",
                        "Generate APA 7th and IEEE information.",
                        "citation"
                    )}

                    ${dashboardTool(
                        "📝",
                        "Paper Write-up",
                        "Generate focused academic sections.",
                        "writeup"
                    )}

                </div>

            </section>


            <!-- RECENT CONTENT -->

            <div class="
                grid
                grid-cols-1
                xl:grid-cols-2
                gap-5
            ">


                <section class="
                    bg-white
                    dark:bg-slate-800
                    border
                    border-slate-200
                    dark:border-slate-700
                    rounded-2xl
                    shadow-sm
                    overflow-hidden
                ">

                    <div class="
                        px-5
                        py-4
                        border-b
                        border-slate-100
                        dark:border-slate-700
                        flex
                        items-center
                        justify-between
                    ">

                        <div>

                            <h3 class="
                                font-semibold
                                text-slate-900
                                dark:text-white
                            ">
                                Recent Projects
                            </h3>

                            <p class="
                                text-[11px]
                                text-slate-400
                                mt-0.5
                            ">
                                Your latest research work
                            </p>

                        </div>

                        <button
                            type="button"
                            data-dashboard-projects
                            class="
                                text-xs
                                font-semibold
                                text-indigo-600
                                dark:text-indigo-300
                            "
                        >
                            View all →
                        </button>

                    </div>


                    <div id="dashboardProjects">

                        ${
                            projects.length
                                ? projects
                                    .slice(0, 5)
                                    .map(
                                        project => `
                                            <div class="
                                                px-5
                                                py-4
                                                border-b
                                                border-slate-100
                                                dark:border-slate-700
                                            ">

                                                <div class="
                                                    font-medium
                                                    text-sm
                                                    text-slate-800
                                                    dark:text-white
                                                ">
                                                    ${escapeHtml(
                                                        project.title ||
                                                        project.name ||
                                                        "Research Project"
                                                    )}
                                                </div>

                                                <div class="
                                                    text-xs
                                                    text-slate-400
                                                    mt-1
                                                ">
                                                    ${escapeHtml(
                                                        project.description ||
                                                        "Research workspace"
                                                    )}
                                                </div>

                                            </div>
                                        `
                                    )
                                    .join("")
                                :
                                `
                                    <div class="
                                        p-8
                                        text-center
                                        text-sm
                                        text-slate-400
                                    ">
                                        No projects loaded yet.
                                    </div>
                                `
                        }

                    </div>

                </section>


                <section class="
                    bg-white
                    dark:bg-slate-800
                    border
                    border-slate-200
                    dark:border-slate-700
                    rounded-2xl
                    shadow-sm
                    overflow-hidden
                ">

                    <div class="
                        px-5
                        py-4
                        border-b
                        border-slate-100
                        dark:border-slate-700
                        flex
                        items-center
                        justify-between
                    ">

                        <div>

                            <h3 class="
                                font-semibold
                                text-slate-900
                                dark:text-white
                            ">
                                Research Library
                            </h3>

                            <p class="
                                text-[11px]
                                text-slate-400
                                mt-0.5
                            ">
                                Recently available papers
                            </p>

                        </div>

                        <button
                            type="button"
                            data-dashboard-library
                            class="
                                text-xs
                                font-semibold
                                text-indigo-600
                                dark:text-indigo-300
                            "
                        >
                            View all →
                        </button>

                    </div>


                    <div
                        id="dashboardPapers"
                        class="p-4"
                    >

                        ${
                            papers.length
                                ? `
                                    <div class="
                                        space-y-2
                                    ">

                                        ${
                                            papers
                                                .slice(0, 5)
                                                .map(
                                                    paper => `
                                                        <div class="
                                                            flex
                                                            items-center
                                                            gap-3
                                                            p-3
                                                            rounded-xl
                                                            border
                                                            border-slate-100
                                                            dark:border-slate-700
                                                        ">

                                                            <div class="
                                                                w-9
                                                                h-9
                                                                rounded-lg
                                                                bg-red-50
                                                                text-red-600
                                                                flex
                                                                items-center
                                                                justify-center
                                                            ">
                                                                PDF
                                                            </div>

                                                            <div class="
                                                                min-w-0
                                                                flex-1
                                                            ">

                                                                <div class="
                                                                    text-sm
                                                                    font-medium
                                                                    truncate
                                                                ">
                                                                    ${escapeHtml(
                                                                        getPaperTitle(
                                                                            paper
                                                                        )
                                                                    )}
                                                                </div>

                                                                <div class="
                                                                    text-[10px]
                                                                    text-slate-400
                                                                    mt-1
                                                                ">
                                                                    Paper ID:
                                                                    ${getPaperId(
                                                                        paper
                                                                    ) ?? "—"}
                                                                </div>

                                                            </div>

                                                        </div>
                                                    `
                                                )
                                                .join("")
                                        }

                                    </div>
                                `
                                :
                                `
                                    <div class="
                                        p-8
                                        text-center
                                        text-sm
                                        text-slate-400
                                    ">
                                        No papers loaded yet.
                                    </div>
                                `
                        }

                    </div>

                </section>

            </div>

        </div>
    `;


    container
        .querySelectorAll(
            "[data-dashboard-projects]"
        )
        .forEach(
            button => {

                button.addEventListener(
                    "click",
                    () => {
                        loadPage(
                            "projects"
                        );
                    }
                );
            }
        );


    container
        .querySelectorAll(
            "[data-dashboard-library]"
        )
        .forEach(
            button => {

                button.addEventListener(
                    "click",
                    () => {
                        loadPage(
                            "upload"
                        );
                    }
                );
            }
        );


    container
        .querySelectorAll(
            "[data-dashboard-tool]"
        )
        .forEach(
            button => {

                button.addEventListener(
                    "click",
                    () => {

                        const page =
                            button.dataset.dashboardTool;

                        if (
                            PAGE_TITLES[page]
                        ) {
                            loadPage(page);
                        }
                    }
                );
            }
        );
}


function dashboardMetric(
    icon,
    value,
    title
) {

    return `
        <div class="
            bg-white
            dark:bg-slate-800
            border
            border-slate-200
            dark:border-slate-700
            rounded-2xl
            p-4
            shadow-sm
        ">

            <div class="
                w-9
                h-9
                rounded-xl
                bg-indigo-50
                dark:bg-indigo-900/30
                flex
                items-center
                justify-center
                text-lg
            ">
                ${icon}
            </div>

            <div class="
                text-2xl
                font-bold
                mt-3
                text-slate-900
                dark:text-white
            ">
                ${escapeHtml(value)}
            </div>

            <div class="
                text-xs
                text-slate-500
                dark:text-slate-400
                mt-1
            ">
                ${escapeHtml(title)}
            </div>

        </div>
    `;
}


function dashboardTool(
    icon,
    title,
    description,
    page
) {

    return `
        <button
            type="button"
            data-dashboard-tool="${page}"
            class="
                text-left
                p-5
                rounded-xl
                border
                border-slate-200
                dark:border-slate-700
                hover:border-indigo-300
                hover:bg-indigo-50/30
                dark:hover:bg-indigo-900/20
                transition
            "
        >

            <div class="text-2xl">
                ${icon}
            </div>

            <div class="
                font-semibold
                text-sm
                mt-3
                text-slate-800
                dark:text-white
            ">
                ${escapeHtml(title)}
            </div>

            <div class="
                text-xs
                text-slate-400
                mt-1
                leading-5
            ">
                ${escapeHtml(description)}
            </div>

        </button>
    `;
}


function updateDashboardData(
    container
) {

    if (
        currentPage !==
        "dashboard"
    ) {
        return;
    }

    /*
     * Re-render dashboard after background data
     * arrives. This happens after the initial UI
     * is already visible.
     */

    renderDashboard(
        container
    );
}


/* ============================================================
   PAPER SELECTOR
   ============================================================ */

async function getPapersForSelector() {

    if (
        papersCache.length
    ) {
        return papersCache;
    }

    return await loadPapersBackground();
}


function renderPaperSelectorHTML(
    papers,
    className = "paper-selection"
) {

    if (
        !papers.length
    ) {

        return emptyState(
            "📄",
            "No papers available",
            "Upload and process a research paper first."
        );
    }


    return `
        <div
            class="
                space-y-2
                max-h-[430px]
                overflow-y-auto
                pr-1
            "
        >

            ${
                papers
                    .map(
                        paper => {

                            const id =
                                getPaperId(
                                    paper
                                );

                            if (
                                id === null
                            ) {
                                return "";
                            }

                            return `
                                <label class="
                                    flex
                                    items-start
                                    gap-3
                                    p-3
                                    rounded-xl
                                    border
                                    border-slate-200
                                    dark:border-slate-700
                                    hover:border-indigo-400
                                    cursor-pointer
                                    transition
                                ">

                                    <input
                                        type="checkbox"
                                        value="${id}"
                                        class="${className}
                                            mt-1
                                            w-4
                                            h-4
                                            accent-indigo-600
                                        "
                                    >

                                    <div class="min-w-0">

                                        <div class="
                                            text-sm
                                            font-medium
                                            text-slate-800
                                            dark:text-white
                                        ">
                                            ${escapeHtml(
                                                getPaperTitle(
                                                    paper
                                                )
                                            )}
                                        </div>

                                        ${
                                            getPaperAuthors(
                                                paper
                                            )
                                                ? `
                                                    <div class="
                                                        text-xs
                                                        text-slate-400
                                                        mt-1
                                                    ">
                                                        ${escapeHtml(
                                                            getPaperAuthors(
                                                                paper
                                                            )
                                                        )}
                                                    </div>
                                                `
                                                : ""
                                        }

                                    </div>

                                </label>
                            `;
                        }
                    )
                    .join("")
            }

        </div>
    `;
}


function selectedIds(
    container,
    className = "paper-selection"
) {

    if (!container) {
        return [];
    }

    return Array.from(
        container.querySelectorAll(
            `.${className}:checked`
        )
    )
        .map(
            input =>
                Number(
                    input.value
                )
        )
        .filter(
            Number.isFinite
        );
}


function bindPaperSelection(
    container,
    counter,
    button,
    minimum = 1
) {

    if (
        !container
    ) {
        return;
    }

    function update() {

        const ids =
            selectedIds(
                container
            );

        if (counter) {

            counter.textContent =
                `${ids.length} selected`;
        }

        if (button) {

            button.disabled =
                ids.length <
                minimum;
        }
    }


    container.addEventListener(
        "change",
        event => {

            if (
                !event.target.classList.contains(
                    "paper-selection"
                )
            ) {
                return;
            }

            const checked =
                container.querySelectorAll(
                    ".paper-selection:checked"
                );

            if (
                checked.length >
                MAX_PAPERS
            ) {

                event.target.checked =
                    false;

                showToast(
                    "Maximum 10 papers can be selected.",
                    "error"
                );
            }

            update();
        }
    );


    update();
}


/* ============================================================
   PROJECTS
   ============================================================ */

async function renderProjects(
    container
) {

    const projects =
        await loadProjectsBackground();


    container.innerHTML = `

        <div class="
            max-w-5xl
            mx-auto
            space-y-5
        ">

            <div>

                <h2 class="
                    text-xl
                    font-bold
                    text-slate-900
                    dark:text-white
                ">
                    My Projects
                </h2>

                <p class="
                    text-sm
                    text-slate-500
                    dark:text-slate-400
                    mt-1
                ">
                    Organize your research work.
                </p>

            </div>


            <section class="
                bg-white
                dark:bg-slate-800
                border
                border-slate-200
                dark:border-slate-700
                rounded-2xl
                p-5
            ">

                ${
                    projects.length
                        ? projects
                            .map(
                                project => `
                                    <div class="
                                        py-4
                                        border-b
                                        border-slate-100
                                        dark:border-slate-700
                                        last:border-0
                                    ">

                                        <div class="
                                            font-medium
                                            text-sm
                                        ">
                                            ${escapeHtml(
                                                project.title ||
                                                project.name ||
                                                "Research Project"
                                            )}
                                        </div>

                                        <div class="
                                            text-xs
                                            text-slate-400
                                            mt-1
                                        ">
                                            ${escapeHtml(
                                                project.description ||
                                                "Research workspace"
                                            )}
                                        </div>

                                    </div>
                                `
                            )
                            .join("")
                        :
                        emptyState(
                            "📁",
                            "No projects yet",
                            "Create your first research project."
                        )
                }

            </section>

        </div>
    `;
}


/* ============================================================
   LIBRARY
   ============================================================ */

async function renderLibrary(
    container
) {

    const papers =
        await loadPapersBackground();


    container.innerHTML = `

        <div class="
            max-w-6xl
            mx-auto
            space-y-5
        ">

            <div>

                <h2 class="
                    text-xl
                    font-bold
                ">
                    Library & Papers
                </h2>

                <p class="
                    text-sm
                    text-slate-500
                    dark:text-slate-400
                    mt-1
                ">
                    Upload and manage your research papers.
                </p>

            </div>


            <section class="
                bg-white
                dark:bg-slate-800
                border
                border-slate-200
                dark:border-slate-700
                rounded-2xl
                p-6
            ">

                <label class="
                    block
                    border-2
                    border-dashed
                    border-slate-300
                    dark:border-slate-600
                    rounded-2xl
                    p-10
                    text-center
                    cursor-pointer
                    hover:border-indigo-400
                ">

                    <input
                        id="paperUploadInput"
                        type="file"
                        accept=".pdf"
                        class="hidden"
                    >

                    <div class="
                        text-4xl
                    ">
                        📄
                    </div>

                    <div class="
                        font-medium
                        mt-3
                    ">
                        Upload a research paper
                    </div>

                    <div class="
                        text-xs
                        text-slate-400
                        mt-1
                    ">
                        PDF files only
                    </div>

                </label>

                <div
                    id="uploadMessage"
                    class="mt-4 text-sm"
                ></div>

            </section>


            <section class="
                bg-white
                dark:bg-slate-800
                border
                border-slate-200
                dark:border-slate-700
                rounded-2xl
                overflow-hidden
            ">

                <div class="
                    px-5
                    py-4
                    border-b
                    border-slate-200
                    dark:border-slate-700
                ">

                    <h3 class="font-semibold">
                        Uploaded Papers
                    </h3>

                </div>


                <div class="p-5">

                    ${
                        papers.length
                            ? papers
                                .map(
                                    paper => `
                                        <div class="
                                            flex
                                            items-center
                                            justify-between
                                            gap-4
                                            py-4
                                            border-b
                                            border-slate-100
                                            dark:border-slate-700
                                            last:border-0
                                        ">

                                            <div class="min-w-0">

                                                <div class="
                                                    font-medium
                                                    text-sm
                                                    truncate
                                                ">
                                                    ${escapeHtml(
                                                        getPaperTitle(
                                                            paper
                                                        )
                                                    )}
                                                </div>

                                                <div class="
                                                    text-xs
                                                    text-slate-400
                                                    mt-1
                                                ">
                                                    ID:
                                                    ${getPaperId(
                                                        paper
                                                    ) ?? "—"}

                                                    ·

                                                    Status:
                                                    ${escapeHtml(
                                                        paper.status ||
                                                        "unknown"
                                                    )}
                                                </div>

                                            </div>


                                            <div class="
                                                flex
                                                gap-2
                                            ">

                                                ${
                                                    paper.status !==
                                                    "indexed"
                                                        ? `
                                                            <button
                                                                type="button"
                                                                data-process-paper="${getPaperId(
                                                                    paper
                                                                )}"
                                                                class="
                                                                    px-3
                                                                    py-2
                                                                    rounded-lg
                                                                    bg-indigo-600
                                                                    text-white
                                                                    text-xs
                                                                "
                                                            >
                                                                Process
                                                            </button>
                                                        `
                                                        : ""
                                                }


                                                <button
                                                    type="button"
                                                    data-delete-paper="${getPaperId(
                                                        paper
                                                    )}"
                                                    class="
                                                        px-3
                                                        py-2
                                                        rounded-lg
                                                        bg-red-600
                                                        text-white
                                                        text-xs
                                                    "
                                                >
                                                    Delete
                                                </button>

                                            </div>

                                        </div>
                                    `
                                )
                                .join("")
                            :
                            emptyState(
                                "📚",
                                "No papers uploaded",
                                "Upload a PDF to begin."
                            )
                    }

                </div>

            </section>

        </div>
    `;


    const input =
        document.getElementById(
            "paperUploadInput"
        );


    input?.addEventListener(
        "change",
        async event => {

            const file =
                event.target.files?.[0];

            if (!file) {
                return;
            }

            const message =
                document.getElementById(
                    "uploadMessage"
                );


            try {

                message.className =
                    "mt-4 text-sm text-indigo-600";

                message.textContent =
                    "Uploading paper...";


                await window.uploadPaper(
                    file,
                    file.name.replace(
                        /\.pdf$/i,
                        ""
                    )
                );


                papersCache = [];


                message.className =
                    "mt-4 text-sm text-emerald-600";

                message.textContent =
                    "Paper uploaded successfully.";


                await renderLibrary(
                    container
                );


            } catch (error) {

                message.className =
                    "mt-4 text-sm text-red-600";

                message.textContent =
                    error.message;
            }
        }
    );


    container
        .querySelectorAll(
            "[data-process-paper]"
        )
        .forEach(
            button => {

                button.addEventListener(
                    "click",
                    async () => {

                        const id =
                            Number(
                                button.dataset.processPaper
                            );

                        try {

                            button.disabled =
                                true;

                            button.textContent =
                                "Processing...";


                            await window.processPaper(
                                id
                            );


                            papersCache = [];


                            showToast(
                                "Paper processed successfully.",
                                "success"
                            );


                            await renderLibrary(
                                container
                            );


                        } catch (error) {

                            button.disabled =
                                false;

                            button.textContent =
                                "Process";

                            showToast(
                                error.message,
                                "error"
                            );
                        }
                    }
                );
            }
        );


    container
        .querySelectorAll(
            "[data-delete-paper]"
        )
        .forEach(
            button => {

                button.addEventListener(
                    "click",
                    async () => {

                        const id =
                            Number(
                                button.dataset.deletePaper
                            );


                        if (
                            !confirm(
                                "Delete this paper?"
                            )
                        ) {
                            return;
                        }


                        try {

                            await window.deletePaper(
                                id
                            );

                            papersCache = [];


                            showToast(
                                "Paper deleted successfully.",
                                "success"
                            );


                            await renderLibrary(
                                container
                            );


                        } catch (error) {

                            showToast(
                                error.message,
                                "error"
                            );
                        }
                    }
                );
            }
        );
}


/* ============================================================
   AI CHAT
   ============================================================ */

async function renderChat(
    container
) {

    const papers =
        await getPapersForSelector();


    container.innerHTML = `

        <div class="
            max-w-6xl
            mx-auto
            space-y-5
        ">

            <div>

                <h2 class="text-xl font-bold">
                    AI Chat
                </h2>

                <p class="
                    text-sm
                    text-slate-500
                    dark:text-slate-400
                    mt-1
                ">
                    Ask evidence-grounded questions about your papers.
                </p>

            </div>


            <section class="
                bg-white
                dark:bg-slate-800
                border
                border-slate-200
                dark:border-slate-700
                rounded-2xl
                p-5
            ">

                <label class="
                    text-sm
                    font-semibold
                ">
                    Select paper
                </label>


                <select
                    id="chatPaper"
                    class="
                        w-full
                        mt-2
                        p-3
                        rounded-xl
                        border
                        border-slate-300
                        dark:border-slate-600
                        bg-white
                        dark:bg-slate-900
                    "
                >

                    ${
                        papers
                            .map(
                                paper => `
                                    <option
                                        value="${getPaperId(
                                            paper
                                        )}"
                                    >
                                        ${escapeHtml(
                                            getPaperTitle(
                                                paper
                                            )
                                        )}
                                    </option>
                                `
                            )
                            .join("")
                    }

                </select>


                <label class="
                    block
                    text-sm
                    font-semibold
                    mt-5
                ">
                    Question
                </label>


                <textarea
                    id="chatQuestion"
                    rows="5"
                    class="
                        w-full
                        mt-2
                        p-3
                        rounded-xl
                        border
                        border-slate-300
                        dark:border-slate-600
                        bg-white
                        dark:bg-slate-900
                    "
                    placeholder="Ask about the methodology, results, limitations, contribution or findings..."
                ></textarea>


                <div class="
                    flex
                    justify-end
                    mt-3
                ">

                    <button
                        id="chatSend"
                        type="button"
                        class="
                            px-5
                            py-2.5
                            rounded-xl
                            bg-indigo-600
                            text-white
                            font-semibold
                            text-sm
                        "
                    >
                        Ask ResearchGPT
                    </button>

                </div>

            </section>


            <section
                id="chatResult"
                class="
                    bg-white
                    dark:bg-slate-800
                    border
                    border-slate-200
                    dark:border-slate-700
                    rounded-2xl
                    p-6
                "
            >
                ${emptyState(
                    "💬",
                    "Ready",
                    "Your answer will appear here."
                )}
            </section>

        </div>
    `;


    const button =
        document.getElementById(
            "chatSend"
        );


    button?.addEventListener(
        "click",
        async () => {

            const question =
                document
                    .getElementById(
                        "chatQuestion"
                    )
                    .value
                    .trim();


            const paperId =
                Number(
                    document
                        .getElementById(
                            "chatPaper"
                        )
                        .value
                );


            const result =
                document.getElementById(
                    "chatResult"
                );


            if (!question) {

                showToast(
                    "Please enter a question.",
                    "error"
                );

                return;
            }


            button.disabled =
                true;

            button.textContent =
                "Thinking...";


            setLoading(
                result,
                "Retrieving evidence and generating your answer..."
            );


            try {

                const response =
                    await timeoutPromise(
                        window.chatWithPaper(
                            question,
                            paperId
                        ),
                        60000,
                        "AI response timed out. Please try again."
                    );


                result.innerHTML =
                    formatGeneratedText(
                        extractResponseText(
                            response
                        )
                    );


            } catch (error) {

                result.innerHTML = `
                    <div class="
                        p-5
                        rounded-xl
                        bg-red-50
                        dark:bg-red-950/30
                        text-red-600
                    ">
                        ${escapeHtml(
                            error.message
                        )}
                    </div>
                `;

            } finally {

                button.disabled =
                    false;

                button.textContent =
                    "Ask ResearchGPT";
            }
        }
    );
}


/* ============================================================
   MULTI-PAPER PAGE
   ============================================================ */

async function renderMultiPaperTool(
    container,
    type
) {

    const papers =
        await getPapersForSelector();


    const config = {

        literature: {
            icon: "📚",
            title: "Literature Review",
            description:
                "Synthesize the selected research papers into a coherent academic review.",
            button:
                "Generate Literature Review",
            minimum: 1,
            api:
                "generateLiteratureReview"
        },

        "research-gap": {
            icon: "🔍",
            title: "Research Gap",
            description:
                "Identify limitations, unresolved problems and evidence-supported research opportunities.",
            button:
                "Generate Research Gap",
            minimum: 1,
            api:
                "generateResearchGap"
        },

        comparison: {
            icon: "⚖️",
            title: "Paper Comparison",
            description:
                "Compare objectives, methodology, datasets, models, results, contributions and limitations.",
            button:
                "Compare Papers",
            minimum: 2,
            api:
                "comparePapers"
        },

        citation: {
            icon: "📑",
            title: "Citation Manager",
            description:
                "Generate citation-ready academic information for the selected papers.",
            button:
                "Generate Citation Analysis",
            minimum: 1,
            api:
                "generateCitationManager"
        }

    }[type];


    if (!config) {

        container.innerHTML =
            emptyState(
                "⚠️",
                "Tool unavailable",
                "This research tool is not configured."
            );

        return;
    }


    container.innerHTML = `

        <div class="
            max-w-6xl
            mx-auto
            space-y-5
        ">

            <div>

                <h2 class="text-xl font-bold">
                    ${config.icon}
                    ${config.title}
                </h2>

                <p class="
                    text-sm
                    text-slate-500
                    dark:text-slate-400
                    mt-1
                ">
                    ${config.description}
                </p>

            </div>


            <section class="
                bg-white
                dark:bg-slate-800
                border
                border-slate-200
                dark:border-slate-700
                rounded-2xl
                p-5
            ">

                <div class="
                    flex
                    items-center
                    justify-between
                ">

                    <div class="
                        text-sm
                        font-semibold
                    ">
                        Select source papers
                    </div>

                    <span
                        id="multiPaperCount"
                        class="
                            text-xs
                            text-slate-400
                        "
                    >
                        0 selected
                    </span>

                </div>


                <div
                    id="multiPaperSelector"
                    class="mt-4"
                >

                    ${
                        renderPaperSelectorHTML(
                            papers
                        )
                    }

                </div>


                <div class="
                    flex
                    justify-end
                    mt-4
                ">

                    <button
                        id="multiGenerateButton"
                        type="button"
                        disabled
                        class="
                            px-5
                            py-2.5
                            rounded-xl
                            bg-indigo-600
                            hover:bg-indigo-700
                            text-white
                            font-semibold
                            text-sm
                            disabled:opacity-40
                            disabled:cursor-not-allowed
                        "
                    >
                        ${config.button}
                    </button>

                </div>

            </section>


            <section
                id="multiResult"
                class="
                    bg-white
                    dark:bg-slate-800
                    border
                    border-slate-200
                    dark:border-slate-700
                    rounded-2xl
                    p-6
                "
            >

                ${emptyState(
                    config.icon,
                    "Ready for analysis",
                    "Select your papers and generate the result."
                )}

            </section>

        </div>
    `;


    const selector =
        document.getElementById(
            "multiPaperSelector"
        );

    const counter =
        document.getElementById(
            "multiPaperCount"
        );

    const button =
        document.getElementById(
            "multiGenerateButton"
        );

    const result =
        document.getElementById(
            "multiResult"
        );


    bindPaperSelection(
        selector,
        counter,
        button,
        config.minimum
    );


    button?.addEventListener(
        "click",
        async () => {

            const ids =
                selectedIds(
                    selector
                );


            if (
                ids.length <
                config.minimum
            ) {

                showToast(
                    config.minimum === 2
                        ? "Select at least 2 papers for comparison."
                        : "Select at least one paper.",
                    "error"
                );

                return;
            }


            const apiFunction =
                window[
                    config.api
                ];


            if (
                typeof apiFunction !==
                "function"
            ) {

                showToast(
                    `${config.title} API is not available.`,
                    "error"
                );

                return;
            }


            button.disabled =
                true;

            button.textContent =
                "Generating...";


            setLoading(
                result,
                "Retrieving paper evidence and generating your analysis..."
            );


            try {

                const response =
                    await timeoutPromise(
                        apiFunction(
                            ids
                        ),
                        60000,
                        "The analysis request timed out. Please try again."
                    );


                const sourceNames =
                    papers
                        .filter(
                            paper =>
                                ids.includes(
                                    getPaperId(
                                        paper
                                    )
                                )
                        )
                        .map(
                            paper =>
                                getPaperTitle(
                                    paper
                                )
                        );


                result.innerHTML = `

                    <div class="
                        space-y-5
                    ">

                        <div class="
                            rounded-xl
                            bg-indigo-50
                            dark:bg-indigo-950/30
                            border
                            border-indigo-100
                            dark:border-indigo-900
                            p-4
                        ">

                            <div class="
                                text-xs
                                font-semibold
                                uppercase
                                tracking-wide
                                text-indigo-600
                            ">
                                Source Papers
                            </div>

                            <div class="
                                text-sm
                                mt-2
                                leading-6
                            ">

                                ${
                                    sourceNames
                                        .map(
                                            name => `
                                                <div>
                                                    •
                                                    ${escapeHtml(
                                                        name
                                                    )}
                                                </div>
                                            `
                                        )
                                        .join("")
                                }

                            </div>

                        </div>


                        ${formatGeneratedText(
                            extractResponseText(
                                response
                            )
                        )}

                    </div>

                `;


            } catch (error) {

                result.innerHTML = `
                    <div class="
                        p-5
                        rounded-xl
                        bg-red-50
                        dark:bg-red-950/30
                        text-red-600
                    ">
                        <strong>
                            ${escapeHtml(
                                config.title
                            )}
                            failed
                        </strong>

                        <div class="mt-2">
                            ${escapeHtml(
                                error.message
                            )}
                        </div>
                    </div>
                `;

            } finally {

                button.disabled =
                    false;

                button.textContent =
                    config.button;
            }
        }
    );
}


/* ============================================================
   TOOL ALIASES
   ============================================================ */

async function renderLiterature(
    container
) {

    return renderMultiPaperTool(
        container,
        "literature"
    );
}


async function renderResearchGap(
    container
) {

    return renderMultiPaperTool(
        container,
        "research-gap"
    );
}


async function renderComparison(
    container
) {

    return renderMultiPaperTool(
        container,
        "comparison"
    );
}


async function renderCitationManager(
    container
) {

    return renderMultiPaperTool(
        container,
        "citation"
    );
}


/* ============================================================
   PAPER WRITE-UP
   ============================================================ */

async function renderPaperWriteup(
    container
) {

    const papers =
        await getPapersForSelector();


    container.innerHTML = `

        <div class="
            max-w-6xl
            mx-auto
            space-y-5
        ">

            <div>

                <h2 class="text-xl font-bold">
                    Paper Write-up
                </h2>

                <p class="
                    text-sm
                    text-slate-500
                    dark:text-slate-400
                    mt-1
                ">
                    Generate focused academic writing grounded in your selected papers.
                </p>

            </div>


            <section class="
                bg-white
                dark:bg-slate-800
                border
                border-slate-200
                dark:border-slate-700
                rounded-2xl
                p-5
            ">

                <div class="
                    grid
                    md:grid-cols-2
                    gap-4
                ">

                    <div>

                        <label class="
                            text-sm
                            font-semibold
                        ">
                            Section
                        </label>

                        <select
                            id="writeupType"
                            class="
                                w-full
                                mt-2
                                p-3
                                rounded-xl
                                border
                                border-slate-300
                                dark:border-slate-600
                                bg-white
                                dark:bg-slate-900
                            "
                        >

                            ${
                                [
                                    "abstract",
                                    "introduction",
                                    "literature_review",
                                    "methodology",
                                    "results",
                                    "discussion",
                                    "conclusion",
                                    "full_paper"
                                ]
                                    .map(
                                        value => `
                                            <option
                                                value="${value}"
                                            >
                                                ${value
                                                    .replace(
                                                        /_/g,
                                                        " "
                                                    )
                                                    .replace(
                                                        /\b\w/g,
                                                        c =>
                                                            c.toUpperCase()
                                                    )}
                                            </option>
                                        `
                                    )
                                    .join("")
                            }

                        </select>

                    </div>


                    <div>

                        <label class="
                            text-sm
                            font-semibold
                        ">
                            Research Topic
                        </label>

                        <input
                            id="writeupResearchTopic"
                            type="text"
                            class="
                                w-full
                                mt-2
                                p-3
                                rounded-xl
                                border
                                border-slate-300
                                dark:border-slate-600
                                bg-white
                                dark:bg-slate-900
                            "
                            placeholder="e.g. AI for echocardiography"
                        >

                    </div>

                </div>


                <label class="
                    block
                    text-sm
                    font-semibold
                    mt-4
                ">
                    Instructions
                </label>


                <textarea
                    id="writeupInstructions"
                    rows="4"
                    class="
                        w-full
                        mt-2
                        p-3
                        rounded-xl
                        border
                        border-slate-300
                        dark:border-slate-600
                        bg-white
                        dark:bg-slate-900
                    "
                    placeholder="Specify academic style, emphasis, structure or terminology."
                ></textarea>

            </section>


            <section class="
                bg-white
                dark:bg-slate-800
                border
                border-slate-200
                dark:border-slate-700
                rounded-2xl
                p-5
            ">

                <div class="
                    flex
                    items-center
                    justify-between
                ">

                    <div class="
                        font-semibold
                        text-sm
                    ">
                        Select source papers
                    </div>

                    <span
                        id="writeupCount"
                        class="
                            text-xs
                            text-slate-400
                        "
                    >
                        0 selected
                    </span>

                </div>


                <div
                    id="writeupPaperSelector"
                    class="mt-4"
                >

                    ${
                        renderPaperSelectorHTML(
                            papers
                        )
                    }

                </div>


                <div class="
                    flex
                    justify-end
                    mt-4
                ">

                    <button
                        id="generateWriteupButton"
                        type="button"
                        disabled
                        class="
                            px-5
                            py-2.5
                            rounded-xl
                            bg-indigo-600
                            hover:bg-indigo-700
                            text-white
                            font-semibold
                            text-sm
                            disabled:opacity-40
                        "
                    >
                        Generate Write-up
                    </button>

                </div>

            </section>


            <section
                id="writeupResult"
                class="
                    bg-white
                    dark:bg-slate-800
                    border
                    border-slate-200
                    dark:border-slate-700
                    rounded-2xl
                    p-6
                "
            >

                ${emptyState(
                    "📝",
                    "Write-up workspace",
                    "Select papers and generate your academic section."
                )}

            </section>

        </div>
    `;


    const selector =
        document.getElementById(
            "writeupPaperSelector"
        );

    const counter =
        document.getElementById(
            "writeupCount"
        );

    const button =
        document.getElementById(
            "generateWriteupButton"
        );

    const result =
        document.getElementById(
            "writeupResult"
        );


    bindPaperSelection(
        selector,
        counter,
        button,
        1
    );


    button?.addEventListener(
        "click",
        async () => {

            const ids =
                selectedIds(
                    selector
                );


            if (!ids.length) {

                showToast(
                    "Select at least one paper.",
                    "error"
                );

                return;
            }


            if (
                typeof window.generatePaperWriteup !==
                "function"
            ) {

                showToast(
                    "Paper Write-up API is not available.",
                    "error"
                );

                return;
            }


            const type =
                document
                    .getElementById(
                        "writeupType"
                    )
                    .value;


            const topic =
                document
                    .getElementById(
                        "writeupResearchTopic"
                    )
                    .value
                    .trim();


            const instructions =
                document
                    .getElementById(
                        "writeupInstructions"
                    )
                    .value
                    .trim();


            button.disabled =
                true;

            button.textContent =
                "Generating...";


            setLoading(
                result,
                "Retrieving paper evidence and preparing the write-up..."
            );


            try {

                const response =
                    await timeoutPromise(
                        window.generatePaperWriteup(
                            ids,
                            type,
                            topic,
                            instructions
                        ),
                        120000,
                        "Write-up generation timed out. Please try again."
                    );


                const text =
                    extractResponseText(
                        response
                    );


                const names =
                    papers
                        .filter(
                            paper =>
                                ids.includes(
                                    getPaperId(
                                        paper
                                    )
                                )
                        )
                        .map(
                            paper =>
                                getPaperTitle(
                                    paper
                                )
                        );


                result.innerHTML = `

                    <div class="space-y-5">

                        <div class="
                            rounded-xl
                            bg-indigo-50
                            dark:bg-indigo-950/30
                            border
                            border-indigo-100
                            dark:border-indigo-900
                            p-4
                        ">

                            <div class="
                                text-xs
                                font-semibold
                                uppercase
                                tracking-wide
                                text-indigo-600
                            ">
                                Source Papers
                            </div>

                            <div class="
                                text-sm
                                mt-2
                                leading-6
                            ">

                                ${
                                    names
                                        .map(
                                            name => `
                                                <div>
                                                    •
                                                    ${escapeHtml(
                                                        name
                                                    )}
                                                </div>
                                            `
                                        )
                                        .join("")
                                }

                            </div>

                        </div>


                        <div class="
                            flex
                            justify-end
                        ">

                            <button
                                id="copyWriteup"
                                type="button"
                                class="
                                    px-4
                                    py-2
                                    rounded-xl
                                    border
                                    border-slate-200
                                    dark:border-slate-600
                                    text-sm
                                "
                            >
                                Copy
                            </button>

                        </div>


                        ${formatGeneratedText(
                            text
                        )}

                    </div>

                `;


                document
                    .getElementById(
                        "copyWriteup"
                    )
                    ?.addEventListener(
                        "click",
                        async () => {

                            try {

                                await navigator
                                    .clipboard
                                    .writeText(
                                        text
                                    );

                                showToast(
                                    "Write-up copied.",
                                    "success"
                                );

                            } catch {

                                showToast(
                                    "Copy failed.",
                                    "error"
                                );
                            }
                        }
                    );


            } catch (error) {

                result.innerHTML = `
                    <div class="
                        p-5
                        rounded-xl
                        bg-red-50
                        text-red-600
                    ">
                        <strong>
                            Paper Write-up failed
                        </strong>

                        <div class="mt-2">
                            ${escapeHtml(
                                error.message
                            )}
                        </div>
                    </div>
                `;

            } finally {

                button.disabled =
                    false;

                button.textContent =
                    "Generate Write-up";
            }
        }
    );
}


/* ============================================================
   SETTINGS
   ============================================================ */

async function renderSettings(
    container
) {

    container.innerHTML = `

        <div class="
            max-w-4xl
            mx-auto
            space-y-5
        ">

            <div>

                <h2 class="
                    text-xl
                    font-bold
                ">
                    Settings
                </h2>

                <p class="
                    text-sm
                    text-slate-500
                    dark:text-slate-400
                    mt-1
                ">
                    ResearchGPT workspace settings.
                </p>

            </div>


            <section class="
                bg-white
                dark:bg-slate-800
                border
                border-slate-200
                dark:border-slate-700
                rounded-2xl
                p-6
            ">

                <h3 class="
                    font-semibold
                ">
                    Research Workspace
                </h3>

                <p class="
                    text-sm
                    text-slate-500
                    dark:text-slate-400
                    mt-2
                    leading-6
                ">
                    ResearchGPT uses evidence retrieval and
                    configured language models through the backend.
                </p>


                <div class="
                    mt-5
                    grid
                    sm:grid-cols-2
                    gap-3
                ">

                    <div class="
                        rounded-xl
                        bg-slate-50
                        dark:bg-slate-900/40
                        p-4
                    ">

                        <div class="
                            text-xs
                            text-slate-400
                        ">
                            Project Name
                        </div>

                        <div class="
                            font-semibold
                            mt-1
                        ">
                            ResearchGPT
                        </div>

                    </div>


                    <div class="
                        rounded-xl
                        bg-slate-50
                        dark:bg-slate-900/40
                        p-4
                    ">

                        <div class="
                            text-xs
                            text-slate-400
                        ">
                            Research Mode
                        </div>

                        <div class="
                            font-semibold
                            mt-1
                        ">
                            Evidence-grounded AI
                        </div>

                    </div>

                </div>

            </section>

        </div>
    `;
}


/* ============================================================
   NAVIGATION EVENTS
   ============================================================ */

function setupNavigation() {

    document.addEventListener(
        "click",
        event => {

            const nav =
                event.target.closest(
                    ".sidebar-item[data-page]"
                );


            if (nav) {

                event.preventDefault();

                const page =
                    nav.dataset.page;

                if (
                    PAGE_TITLES[page]
                ) {

                    loadPage(
                        page
                    );
                }

                closeMobileSidebar();

                return;
            }


            const dashboardButton =
                event.target.closest(
                    "[data-dashboard-page]"
                );


            if (
                dashboardButton
            ) {

                event.preventDefault();

                const page =
                    dashboardButton.dataset
                        .dashboardPage;

                if (
                    PAGE_TITLES[page]
                ) {

                    loadPage(
                        page
                    );
                }
            }
        }
    );
}


/* ============================================================
   MOBILE MENU
   ============================================================ */

function closeMobileSidebar() {

    const sidebar =
        document.getElementById(
            "sidebar"
        );

    const overlay =
        document.getElementById(
            "mobileOverlay"
        );


    sidebar?.classList.add(
        "-translate-x-full"
    );

    overlay?.classList.add(
        "hidden"
    );
}


function setupMobileMenu() {

    const sidebar =
        document.getElementById(
            "sidebar"
        );

    const overlay =
        document.getElementById(
            "mobileOverlay"
        );

    const open =
        document.getElementById(
            "openSidebar"
        );

    const close =
        document.getElementById(
            "closeSidebar"
        );


    function show() {

        sidebar?.classList.remove(
            "-translate-x-full"
        );

        overlay?.classList.remove(
            "hidden"
        );
    }


    open?.addEventListener(
        "click",
        show
    );


    close?.addEventListener(
        "click",
        closeMobileSidebar
    );


    overlay?.addEventListener(
        "click",
        closeMobileSidebar
    );
}


/* ============================================================
   THEME
   ============================================================ */

function setupTheme() {

    const stored =
        localStorage.getItem(
            "researchGPT-theme"
        ) ||
        localStorage.getItem(
            "paperaxiom-theme"
        );


    if (
        stored === "dark"
    ) {

        document.documentElement
            .classList.add(
                "dark"
            );
    }


    const button =
        document.getElementById(
            "themeToggle"
        );


    button?.addEventListener(
        "click",
        () => {

            const dark =
                document.documentElement
                    .classList.toggle(
                        "dark"
                    );


            localStorage.setItem(
                "researchGPT-theme",
                dark
                    ? "dark"
                    : "light"
            );
        }
    );
}


/* ============================================================
   GLOBAL SEARCH
   ============================================================ */

function setupGlobalSearch() {

    const input =
        document.querySelector(
            "[data-global-search]"
        );


    if (!input) {
        return;
    }


    input.addEventListener(
        "keydown",
        event => {

            if (
                event.key !==
                "Enter"
            ) {
                return;
            }


            const query =
                input.value.trim();


            if (!query) {
                return;
            }


            loadPage(
                "chat"
            );


            /*
             * Put query into chat if available.
             */

            setTimeout(
                () => {

                    const textarea =
                        document.getElementById(
                            "chatQuestion"
                        );

                    if (textarea) {
                        textarea.value =
                            query;
                    }
                },
                100
            );
        }
    );
}


/* ============================================================
   AUTH CHECK
   ============================================================ */

function hasAuthenticationToken() {

    if (
        typeof window.getToken !==
        "function"
    ) {
        return true;
    }

    return Boolean(
        window.getToken()
    );
}


/* ============================================================
   START APPLICATION
   ============================================================ */

function startResearchGPT() {

    console.log(
        "ResearchGPT frontend starting..."
    );


    /*
     * IMPORTANT:
     *
     * We do NOT await anything here.
     *
     * The old startup flow waited for:
     *
     * getMe()
     * getPapers()
     * getProjects()
     * loadPage()
     *
     * before making the UI usable.
     *
     * That is exactly what caused the long startup problem.
     */


    if (
        !hasAuthenticationToken()
    ) {

        if (
            !window.location.pathname.endsWith(
                "login.html"
            )
        ) {

            window.location.href =
                "login.html";

            return;
        }
    }


    setupNavigation();

    setupMobileMenu();

    setupTheme();

    setupGlobalSearch();


    /*
     * Remove boot screen immediately.
     */

    document.body.classList.remove(
        "app-booting"
    );

    document.body.classList.add(
        "app-ready"
    );


    /*
     * Render dashboard immediately.
     */

    const initialPage =
        location.hash
            .replace(
                /^#/,
                ""
            ) ||
        "dashboard";


    loadPage(
        PAGE_TITLES[initialPage]
            ? initialPage
            : "dashboard"
    );


    /*
     * Everything below happens in background.
     */

    loadCurrentUserBackground();

    loadPapersBackground();

    loadProjectsBackground();


    console.log(
        "ResearchGPT UI ready."
    );
}


/* ============================================================
   GLOBAL EXPORTS
   ============================================================ */

window.loadPage =
    loadPage;

window.paperAxiomLoadPage =
    loadPage;

window.researchGPTLoadPage =
    loadPage;

window.renderPaperWriteup =
    renderPaperWriteup;

window.renderCitationManager =
    renderCitationManager;

window.renderResearchGap =
    renderResearchGap;

window.renderComparison =
    renderComparison;

window.renderLiterature =
    renderLiterature;

window.getPaperId =
    getPaperId;

window.getPaperTitle =
    getPaperTitle;

window.showToast =
    showToast;


/* ============================================================
   START
   ============================================================ */

if (
    document.readyState ===
    "loading"
) {

    document.addEventListener(
        "DOMContentLoaded",
        startResearchGPT,
        {
            once: true
        }
    );

} else {

    startResearchGPT();
}