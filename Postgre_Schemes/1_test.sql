SELECT * FROM vees.events_clean
ORDER BY event_id ASC LIMIT 100


SELECT 
    county,
    COUNT(*) FILTER (WHERE species_affected = 'Cattle') AS cattle,
    COUNT(*) FILTER (WHERE species_affected = 'Goats') AS goats,
    COUNT(*) FILTER (WHERE species_affected = 'Sheep') AS sheep,
    COUNT(*) FILTER (WHERE species_affected = 'Chicken') AS chicken
FROM vees.events_clean
GROUP BY county
ORDER BY county;