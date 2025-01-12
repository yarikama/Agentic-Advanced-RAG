from textwrap import dedent

constraint_statements = dedent("""
create constraint chunk_id          if not exists for (chunk:__Chunk__)                 require chunk.id            is unique;
create constraint document_id       if not exists for (document:__Document__)           require document.id         is unique;
create constraint community_id      if not exists for (community:__Community__)         require community.id        is unique;
create constraint entity_id         if not exists for (entity:__Entity__)               require entity.id           is unique;
create constraint entity_name       if not exists for (entity:__Entity__)               require entity.name         is unique;
create constraint covariate_title   if not exists for (covariate:__Covariate__)         require covariate.title     is unique;
create constraint relationship_id   if not exists for ()-[relationship:RELATED]->()     require relationship.id     is unique;
""").split(";")

document_statement = dedent("""
// SET DOCUMENT AND ITS PROPERTIES
MERGE (document:__Document__ {id:value.id})
SET document += value {.title, .raw_content}
""")

text_statement = dedent("""
// SET CHUNK AND ITS PROPERTIES
MERGE (chunk:__Chunk__ {id:value.id})
SET chunk += value {.text, .n_tokens}

// ADD RELATIONSHIPS BETWEEN CHUNKS AND DOCUMENTS
WITH chunk, value
UNWIND value.document_ids AS document_id
MATCH (document:__Document__ {id:document_id})
MERGE (chunk)-[:PART_OF]->(document)
""")

entity_statement = dedent("""
// SET ENTITY AND ITS PROPERTIES
MERGE (entity:__Entity__ {id:value.id})
SET entity += value {.human_readable_id, .description, name:replace(value.name,'"','')}

# ADD VECTOR PROPERTY TO ENTITY
WITH entity, value
CALL db.create.setNodeVectorProperty(entity, "description_embedding", value.description_embedding)
CALL apoc.create.addLabels(entity, case when coalesce(value.type,"") = "" then [] else [apoc.text.upperCamelCase(replace(value.type,'"',''))] end) yield node

# ADD RELATIONSHIPS BETWEEN CHUNKS AND ENTITIES
UNWIND value.text_unit_ids AS text_unit_id
MATCH (chunk:__Chunk__ {id:text_unit_id})
MERGE (chunk)-[:HAS_ENTITY]->(entity)
""")

relationship_statement = dedent("""
// SET RELATIONSHIP AND ITS PROPERTIES
MATCH (source_entity:__Entity__ {name:replace(value.source,'"','')})
MATCH (target_entity:__Entity__ {name:replace(value.target,'"','')})
MERGE (source_entity)-[relationship:RELATED {id: value.id}]->(target_entity)
SET relationship += value {.rank, .weight, .human_readable_id, .description, .text_unit_ids}

// ADD VECTOR PROPERTY TO RELATIONSHIP
WITH relationship, value
CALL db.create.setRelationshipVectorProperty(relationship, "description_embedding", value.description_embedding)
RETURN count(*) as createdRelationships
""")


community_statement = dedent("""
// SET COMMUNITY AND ITS PROPERTIES
MERGE (community:__Community__ {id:value.id})
SET community += value {.level, .title}

// ADD RELATIONSHIPS BETWEEN CHUNKS AND COMMUNITIES (Complexity too high)
/*
WITH *
UNWIND value.text_unit_ids as text_unit_id
MATCH (chunk:__Chunk__ {id:text_unit_id})
MERGE (community)-[:HAS_CHUNK]->(chunk)
*/

// SET RELATIONSHIPS BETWEEN ENTITIES AND COMMUNITIES
WITH *
UNWIND value.relationship_ids as relationship_id
MATCH (source_entity:__Entity__)-[:RELATED {id:relationship_id}]->(target_entity:__Entity__)
MERGE (source_entity)-[:IN_COMMUNITY]->(community)
MERGE (target_entity)-[:IN_COMMUNITY]->(community)
RETURN count(distinct community) as createdCommunities
""")

community_report_statement = dedent("""
// SET COMMUNITY REPORT AND ITS PROPERTIES
MERGE (community:__Community__ {id:value.community})
SET community += value {.level, .title, .rank, .rank_explanation, .full_content, .summary}

// ADD VECTOR PROPERTY TO COMMUNITY REPORT
WITH community, value
CALL db.create.setNodeVectorProperty(community, "summary_embedding", value.summary_embedding)

// ADD RELATIONSHIPS BETWEEN COMMUNITIES AND FINDINGS
WITH community, value
UNWIND range(0, size(value.findings)-1) AS finding_idx
WITH community, value, finding_idx, value.findings[finding_idx] as value_finding
MERGE (community)-[:HAS_FINDING]->(finding:Finding {id:finding_idx})
SET finding += value_finding
""")

covariate_statement = dedent("""
MERGE (covariate:__Covariate__ {id:value.id})
SET covariate += apoc.map.clean(value, ["text_unit_id", "document_ids", "n_tokens"], [NULL, ""])
WITH covariate, value
MATCH (chunk:__Chunk__ {id: value.text_unit_id})
MERGE (chunk)-[:HAS_COVARIATE]->(covariate)
""")