local env = std.extVar("__ksonnet/environments");
local params = std.extVar("__ksonnet/params").components["seldon-core"];

local k = import "k.libsonnet";
local core = import "seldon-core/seldon-core/core.libsonnet";

// updatedParams uses the environment namespace if
// the namespace parameter is not explicitly set
local updatedParams = params {
  namespace: if params.namespace == "null" then env.namespace else params.namespace,
};

local name = params.name;
local namespace = updatedParams.namespace;
local withRbac = params.withRbac;
local withApife = params.withApife;
local withAmbassador = params.withAmbassador;

// APIFE
local apifeImage = params.apifeImage;
local apifeServiceType = params.apifeServiceType;
local grpcMaxMessageSize = params.grpcMaxMessageSize;

// Cluster Manager (The CRD Operator)
local operatorImage = params.operatorImage;
local operatorSpringOptsParam = params.operatorSpringOpts;
local operatorSpringOpts = if operatorSpringOptsParam != "null" then operatorSpringOptsParam else "";
local operatorJavaOptsParam = params.operatorJavaOpts;
local operatorJavaOpts = if operatorJavaOptsParam != "null" then operatorJavaOptsParam else "";

// Engine
local engineImage = params.engineImage;

// APIFE
local apife = [
  core.parts(name,namespace).apife(apifeImage, withRbac, grpcMaxMessageSize),
  core.parts(name,namespace).apifeService(apifeServiceType),
];

local rbac = [
  core.parts(name,namespace).rbacServiceAccount(),
  core.parts(name,namespace).rbacClusterRole(),
  core.parts(name,namespace).rbacRole(),
  core.parts(name,namespace).rbacRoleBinding(),  
  core.parts(name,namespace).rbacClusterRoleBinding(),
];


// Core
local coreComponents = [
  core.parts(name,namespace).deploymentOperator(engineImage, operatorImage, operatorSpringOpts, operatorJavaOpts, withRbac),
  core.parts(name,namespace).redisDeployment(),
  core.parts(name,namespace).redisService(),
  core.parts(name,namespace).crd(),
];


//Ambassador
local ambassadorRbac = [
  core.parts(name,namespace).rbacAmbassadorRole(),
  core.parts(name,namespace).rbacAmbassadorRoleBinding(),  
];

local ambassador = [
  core.parts(name,namespace).ambassadorDeployment(),
  core.parts(name,namespace).ambassadorService(),  
];

local l1 = if withRbac == "true" then rbac + coreComponents else coreComponents;
local l2 = if withApife == "true" then l1 + apife else l1;
local l3 = if withAmbassador == "true" && withRbac == "true" then l2 + ambassadorRbac else l2;
local l4 = if withAmbassador == "true" then l3 + ambassador else l3;

l4