% WARSTWOWA WIZUALIZACJA: statyczne=1, dynamiczne=2 + sciezka plannerMPNET

clear; clc; close all;
seed = 100;

robotRadius = 0.10;   % [m] inflacja przeszkod (bufor)
valDist     = 0.01;   % [m] krok walidacji polaczen
maxLearnedStates = 50;

% Granice mapy do wizualizacji
mapW_vis = 4;
mapH_vis = 3;

% Reczne ustawienie start/cel
start = [0.70 2.6 0.0];   % [x y theta]
goal  = [3.50 0.60 0.0];

%% wczytanie sieci
S = load("trainedMPNet_4x3.mat");
if isfield(S,"mpnet")
    mpnet = S.mpnet;
elseif isfield(S,"trained")
    mpnet = mpnetSE2( ...
        Network=S.trained.trainedNetwork, ...
        StateBounds=S.trained.stateBounds, ...
        EncodingSize=S.trained.encodingSize);
else
    error("trainedMPNet_4x3.mat musi zawierac 'mpnet' lub 'trained'.");
end

%generowanie scenariusza
rng(seed,"twister");
sc = genRandomScenarioForMPNet(seed);

if ~isfield(sc,"mapStatic"), error("Brak sc.mapStatic."); end
if ~isfield(sc,"dyn"),       error("Brak sc.dyn."); end

dyn = sc.dyn;

if ~isfield(dyn,"xCenters") || ~isfield(dyn,"yCenters") || ~isfield(dyn,"d") || ~isfield(dyn,"d_thr")
    error("sc.dyn musi zawierac: xCenters, yCenters, d, d_thr.");
end

% Siatka srodkow komorek (uklad swiata)
[Xc, Yc] = meshgrid(dyn.xCenters, dyn.yCenters);
-
% Statyczne: probkowanie mapy statycznej w punktach siatki
pts = [Xc(:) Yc(:)];
occStatVals = getOccupancy(sc.mapStatic, pts);

% Jesli getOccupancy zwraca NaN poza mapa, traktuj jako zajete
occStatVals(isnan(occStatVals)) = 1;

occStat = reshape(occStatVals >= 0.5, size(Xc));
occDyn = (dyn.d >= dyn.d_thr);

Z = zeros(size(occStat), "double");
Z(occStat) = 1;
Z(occDyn)  = 2;  % dynamiczne nadpisuje statyczne (na wierzchu)

dx = dyn.xCenters(2) - dyn.xCenters(1);
dy = dyn.yCenters(2) - dyn.yCenters(1);
if abs(dx-dy) > 1e-9
    error("Oczekiwane kwadratowe komorki siatki: dx == dy.");
end
resCells = 1/dx;

mapW = dyn.xCenters(end) + dx/2;
mapH = dyn.yCenters(end) + dy/2;

mapPlan = binaryOccupancyMap(mapW, mapH, resCells);

occComp = occStat | occDyn;
occPts = [Xc(occComp) Yc(occComp)];
if ~isempty(occPts)
    setOccupancy(mapPlan, occPts, 1);
end

% inflacja
try
    inflate(mapPlan, robotRadius);
catch
    mapPlan = inflate(mapPlan, robotRadius);
end

%% --- SPRAWDZENIE START/CEL (po inflacji) ---
assertInBounds(start(1:2), mapPlan, "Start");
assertInBounds(goal(1:2),  mapPlan, "Goal");

if getOccupancy(mapPlan, start(1:2)) >= 0.5
    error("Start znajduje sie w przeszkodzie (albo zbyt blisko po inflacji). Zmien start lub zmniejsz robotRadius.");
end
if getOccupancy(mapPlan, goal(1:2)) >= 0.5
    error("Goal znajduje sie w przeszkodzie (albo zbyt blisko po inflacji). Zmien cel lub zmniejsz robotRadius.");
end

%% --- WALIDATOR + plannerMPNET ---
xWorldLim = mapPlan.XWorldLimits;
yWorldLim = mapPlan.YWorldLimits;

ss = stateSpaceSE2([xWorldLim(:)'; yWorldLim(:)'; -pi pi]);

sv = validatorOccupancyMap(ss);
sv.Map = mapPlan;
sv.ValidationDistance = valDist;

try
    planner = plannerMPNET(sv, mpnet, MaxLearnedStates=maxLearnedStates);
catch
    planner = plannerMPNET(sv, mpnet);
    try
        planner.MaxLearnedStates = maxLearnedStates;
    catch
    end
end

[pathMP, infoMP] = plan(planner, start, goal);
fprintf("MPNet: IsPathFound=%d\n", getFieldOrDefault(infoMP,"IsPathFound",0));

%warstwy + sciezka 
figure("Color","w","Position",[80 80 1150 520]);
ax = axes; hold(ax,"on"); axis(ax,"equal"); grid(ax,"on");

imagesc(ax, dyn.xCenters, dyn.yCenters, Z);
set(ax,"YDir","normal"); % wazne: os Y w gore (uklad swiata)

%prostokat 4x3
xlim(ax, [0 mapW_vis]);
ylim(ax, [0 mapH_vis]);
hBorder = rectangle(ax, "Position",[0 0 mapW_vis mapH_vis], ...
    "EdgeColor",[0 0 0], "LineWidth", 2.2, "LineStyle","-");

xlabel(ax,"x [m]"); ylabel(ax,"y [m]");
xlim(ax, [-1, mapW + 1]);
ylim(ax, [-1, mapH + 1]);
% Kolory: 0=white, 1=black (static), 2=red (dynamic)
colormap(ax, [1 1 1; 0 0 0; 0.85 0.10 0.10]);
caxis(ax, [0 2]);

% --- UCHWYTY DO LEGENDY ---
% Dummy uchwyty dla przeszkod (do legendy)
hStatic  = plot(ax, nan, nan, "s", "MarkerSize", 10, ...
    "MarkerFaceColor",[0 0 0], "MarkerEdgeColor",[0 0 0]);
hDynamic = plot(ax, nan, nan, "s", "MarkerSize", 10, ...
    "MarkerFaceColor",[0.85 0.10 0.10], "MarkerEdgeColor",[0.85 0.10 0.10]);

% Sciezka MPNet (uchwyt)
hPath = plot(ax, nan, nan, "g-", "LineWidth", 2.4); % domyslnie (jesli brak sciezki)
if ~isempty(pathMP) && ~isempty(pathMP.States)
    set(hPath, "XData", pathMP.States(:,1), "YData", pathMP.States(:,2));
end

hStart = plot(ax, start(1), start(2), "bo", "LineWidth", 2, "MarkerSize", 8);
hGoal  = plot(ax, goal(1),  goal(2),  "rx", "LineWidth", 2, "MarkerSize", 9);

legend(ax, [hStart hGoal hStatic hDynamic hPath], ...
    {"punkt poczatkowy", "punkt zadany (cel)", "przeszkody statyczne", "przeszkody dynamiczne", "sciezka plannerMPNET"}, ...
    "Location","northeastoutside");

%fp
function v = getFieldOrDefault(s, field, def)
    if isstruct(s) && isfield(s, field)
        v = s.(field);
    else
        v = def;
    end
end

function assertInBounds(xy, mapObj, name)
    xL = mapObj.XWorldLimits;
    yL = mapObj.YWorldLimits;
    if xy(1) < xL(1) || xy(1) > xL(2) || xy(2) < yL(1) || xy(2) > yL(2)
        error("%s poza granicami mapy: (%.3f, %.3f).", name, xy(1), xy(2));
    end
end

